# Plan: LiteLLM gateway

## Why

The current `LLMRouter` is a client-side model router: role -> provider+model,
retry, and a single fallback. It works, but it is capped at two providers, has
no cost accounting, no caching, and no persistence, so a restart forgets
everything it learned about quota.

Moving to LiteLLM buys provider breadth (Groq, NVIDIA NIM, Mistral, Cohere,
Cerebras, Together, ...), latency- and usage-aware routing, unlimited fallback
chains, and per-request cost tracking. It also unblocks two things that are
currently impossible: **using Cohere at all**, and **answering "which model is
best for which task"** with numbers instead of impressions.

Verified before writing this plan (litellm 1.97.0):

- all target providers present in `litellm.provider_list` (149 total)
- one `model_name` can span deployments on different providers, ordered
- cost map has 3,054 models **including open-source ones**
  (`groq/openai/gpt-oss-120b`, `cerebras/llama3.1-8b`, `mistral/*`, `command-r`)
- **`nvidia_nim` has only 3 cost entries, all rerankers** -> NIM chat models
  need manual `input_cost_per_token` / `output_cost_per_token`

## Architectural decision: SDK Router first, proxy later

LiteLLM ships two things that are easy to confuse.

| | SDK `litellm.Router` | Proxy server |
|---|---|---|
| Shape | Python object, in-process | HTTP service + Postgres |
| Setup | `pip install litellm` | docker compose, DB, config |
| Persistence | none | request logs, spend, keys |
| Dashboards / virtual keys / budgets | no | yes |
| Guardrails | callbacks | built-in, configurable |

**Start with the SDK Router.** It gets provider coverage and latency routing
with zero infrastructure, and it keeps every call site unchanged. Move to the
proxy in Phase 5 once you want persisted per-task cost and latency history,
which is what the "which model for which task" question ultimately needs.

## Keep your own facade

Do **not** let `litellm` leak into call sites. Four modules call the LLM today:

```
src/router/classifier.py     complete("routing",    ...)
src/agentic/nodes.py         complete("reasoning",  ...)
src/retrieval/hyde.py        complete("reasoning",  ...)
src/generation/generate.py   complete("generation", ...)
```

Keep `LLMRouter.complete(role, messages, *, temperature, max_tokens,
reasoning_effort) -> LLMResponse` exactly as it is and swap only the internals.
Then none of those four files change, the Streamlit UI keeps working, and the
role abstraction (which is genuinely good, and maps 1:1 onto LiteLLM's
`model_name`) survives.

```
        BEFORE                                AFTER
   call site                             call site
       |  complete("reasoning")              |  complete("reasoning")   <- unchanged
       v                                     v
   LLMRouter                             LLMRouter (thin facade)
       |  _targets / _call_with_retry        |  translate -> litellm
       |  _sleep / QuotaState                v
       v                                 litellm.Router
   GroqProvider / GeminiProvider             |  order, cooldowns, strategy, cost
       |                                     v
       v                              groq | cerebras | mistral | nim | cohere | gemini
   2 providers                           n providers
```

---

## Phase 1 - Keys and a provider smoke test

Goal: know which providers actually work with your keys, and how fast each is.

1. Create free accounts and keys: **Cerebras**, **Mistral**, **NVIDIA NIM**,
   **Cohere**. Add to `.env` (`.env.example` gets the empty placeholders).
2. Write `eval/provider_smoke.py`:
   - for each (provider, model) candidate, send one short prompt and one
     RAG-sized prompt (~5k tokens)
   - record: success, latency, output tokens, reasoning tokens if reported
   - print a ranked table
3. **List models from each provider's API before hardcoding names.** Groq's
   Llama family vanished and Gemini 2.5 was retired mid-project; both were
   found only by querying the live API.

Deliverable: a table of provider x model x latency you can point config at.

## Phase 2 - Config schema

Roles become lists of ordered deployments.

```yaml
llm:
  roles:
    routing:
      - {model: cerebras/llama3.1-8b,                   order: 1}
      - {model: groq/openai/gpt-oss-20b,                order: 2}
      - {model: gemini/gemini-3.5-flash-lite,           order: 3}
    reasoning:
      - {model: groq/openai/gpt-oss-120b,               order: 1}
      - {model: mistral/mistral-large-latest,           order: 2}
      - {model: cerebras/llama-3.3-70b,                 order: 3}
    generation:
      - {model: gemini/gemini-3.6-flash,                order: 1}
      - {model: cohere/command-r,                       order: 2}
  router:
    routing_strategy: latency-based-routing
    num_retries: 3
    allowed_fails: 2
    cooldown_time: 60
  # Only needed for models missing from litellm's cost map (e.g. nvidia_nim).
  cost_overrides:
    nvidia_nim/meta/llama-3.1-8b-instruct:
      input_cost_per_token: 0.0
      output_cost_per_token: 0.0
```

Update `src/config.py`: `LLMConfig.roles` becomes
`dict[str, list[Deployment]]`. Keep `retry` for backward compatibility or drop
it once nothing reads it.

## Phase 3 - Rewrite LLMRouter internals

Keep: `complete()` signature, `LLMResponse`, `AllProvidersExhausted`,
`CallOutcome.fell_back` (the UI shows it), `RoleClient`.

Delete: `_targets`, `_call_with_retry`, `_sleep`, `_record_rate_limit`,
`_record_headers`, `QuotaState`, `DEFAULT_FALLBACK_MODELS`, and both provider
adapters once parity is proven.

```python
class LLMRouter:
    def __init__(self, cfg):
        self._router = litellm.Router(
            model_list=[
                {"model_name": role,
                 "litellm_params": {"model": d.model, "order": d.order},
                 "model_info": cfg.cost_overrides.get(d.model, {})}
                for role, deps in cfg.roles.items() for d in deps
            ],
            routing_strategy=cfg.router.routing_strategy,
            num_retries=cfg.router.num_retries,
            allowed_fails=cfg.router.allowed_fails,
            cooldown_time=cfg.router.cooldown_time,
        )
        self._primary = {role: deps[0].model for role, deps in cfg.roles.items()}

    def complete(self, role, messages, *, temperature=0.0,
                 max_tokens=None, reasoning_effort=None) -> LLMResponse:
        try:
            r = self._router.completion(
                model=role,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature, max_tokens=max_tokens,
                **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
            )
        except Exception as exc:
            raise AllProvidersExhausted(f"role={role}: {exc}") from exc

        used = r.model
        self.last_outcome = CallOutcome(
            provider=used.split("/")[0], model=used,
            fell_back=(used != self._primary[role]),
        )
        return LLMResponse(
            text=r.choices[0].message.content or "",
            model=used, provider=used.split("/")[0],
            finish_reason=r.choices[0].finish_reason or "",
            raw=r,
        )
```

Notes:
- **`fell_back` is now derived** by comparing the served model against the
  primary deployment, rather than tracked during retry.
- **`cost` per call** is available via `litellm.completion_cost(r)` or
  `r._hidden_params["response_cost"]`. Add it to `LLMResponse` and surface it
  in `RagResult.cost` so the Streamlit metrics row can show spend per query.
- LiteLLM raises OpenAI-shaped exceptions (`RateLimitError`,
  `AuthenticationError`, ...). Your taxonomy in `src/llm/base.py` can shrink to
  just `AllProvidersExhausted` plus whatever the UI catches.

## Phase 4 - Verify parity, then exploit it

**Parity (do not skip):**

1. Re-run `eval/runner.py --paths route` and compare routing decisions against
   the cached run. Large drift means the classifier is behaving differently on
   a new model, not that the gateway is broken - check which model served it.
2. Re-run `--paths naive` and compare `contains` per stratum against the
   cached pre-migration run. A drop of more than a few points means the new
   generation model is worse, not that the gateway is broken.
3. Confirm the four call sites are untouched (`git diff --stat` should show no
   changes under `classifier.py`, `nodes.py`, `hyde.py`, `generate.py`).

**Then use what you bought:**

- **`routing_strategy: latency-based-routing`** - directly attacks the slowness.
  It measures real response times and prefers the fastest healthy deployment,
  so you stop guessing whether Cerebras beats Groq.
- **Caching.** Eval re-runs currently re-pay for identical calls. Enable
  `litellm.cache` and iterating on metrics becomes nearly free.
- **The actual goal: model x task comparison.** Hold the corpus and questions
  fixed, point one role at a different deployment, re-run, and compare
  quality / latency / cost. That is a real experiment, and it is now a config
  edit rather than a code change.

## Phase 5 - Proxy server (later)

When you want persisted history, budgets and guardrails:

- `docker compose` with Postgres, `litellm --config config.yaml`
- virtual keys per component, per-key budgets
- guardrails (PII, prompt injection, content filters) declared in config
- the app then talks OpenAI-protocol to `http://localhost:4000` and your
  facade collapses to a single base_url change

Do this only after Phase 4 proves the routing itself is right. Otherwise you
are debugging routing and infrastructure at the same time.

---

## Risks

| Risk | Mitigation |
|---|---|
| `reasoning_effort` silently dropped by some providers | assert reasoning-token counts drop in the Phase 1 smoke test |
| NIM chat models have no cost data | `cost_overrides` in config |
| Model names go stale again | Phase 1 lists models from the live API, never from docs |
| Gemini `thinking_config` handling is lost with the adapter | check whether LiteLLM exposes an equivalent; if not, accept the token cost |
| LiteLLM cooldowns behave differently from `QuotaState` | tune `allowed_fails` / `cooldown_time` against a deliberately rate-limited run |

## Order of execution

```
Phase 1  keys + smoke test          <- start here, unblocks everything
Phase 2  config schema
Phase 3  swap internals
Phase 4  parity, then latency routing + caching
Phase 5  proxy server (optional)
```
