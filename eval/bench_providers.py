#!/usr/bin/env python3
"""Benchmark candidate models across providers on two realistic tasks.

Task A is the actual classifier prompt (short in, short out) - this is what
runs on every query, so latency here matters most.
Task B is a RAG-sized prompt (~5k tokens in) - what generation sees.

Records latency, output tokens and reasoning tokens, because a "fast" model
that emits 500 hidden reasoning tokens is not fast for classification.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import get_config  # noqa: E402

get_config()
# litellm's expected names for keys stored under different vars
os.environ.setdefault("NVIDIA_NIM_API_KEY", os.environ.get("NVIDIA_API_KEY", ""))
os.environ.setdefault("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

import litellm  # noqa: E402
from src.router.classifier import SYSTEM_PROMPT  # noqa: E402

litellm.suppress_debug_info = True

CANDIDATES = [
    "groq/openai/gpt-oss-20b",
    "groq/openai/gpt-oss-120b",
    "groq/qwen/qwen3.6-27b",
    "cerebras/gpt-oss-120b",
    "cerebras/gemma-4-31b",
    "mistral/ministral-3b-latest",
    "mistral/ministral-8b-latest",
    "mistral/mistral-small-latest",
    "mistral/mistral-medium-latest",
    "nvidia_nim/meta/llama-3.1-8b-instruct",
    "nvidia_nim/meta/llama-3.3-70b-instruct",
    "nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2",
    "nvidia_nim/mistralai/mistral-7b-instruct-v0.3",
    "gemini/gemini-3.6-flash",
    "gemini/gemini-3.5-flash-lite",
]

CLASSIFY_Q = ("What region of Afghanistan borders Pewar, Pakistan and was home "
              "to the Safi tribe?")
BIG_CONTEXT = "The paragraph describes a notable historical event. " * 700


def run(model: str, messages: list[dict], max_tokens: int) -> dict:
    t = time.time()
    try:
        r = litellm.completion(model=model, messages=messages, temperature=0,
                               max_tokens=max_tokens, timeout=90)
        el = time.time() - t
        u = r.usage
        det = getattr(u, "completion_tokens_details", None)
        return {
            "ok": True, "s": round(el, 2),
            "in": getattr(u, "prompt_tokens", None),
            "out": getattr(u, "completion_tokens", None),
            "reasoning": getattr(det, "reasoning_tokens", None) if det else None,
            "text": (r.choices[0].message.content or "")[:60],
            "cost": r._hidden_params.get("response_cost"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "s": round(time.time() - t, 2),
                "err": f"{type(exc).__name__}: {str(exc)[:110]}"}


def main() -> int:
    out = {}
    for model in CANDIDATES:
        a = run(model, [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": CLASSIFY_Q}], 512)
        b = ({"ok": False, "err": "skipped (task A failed)"} if not a["ok"] else
             run(model, [{"role": "system", "content": "Answer from the context."},
                         {"role": "user", "content": BIG_CONTEXT + "\n\nQ: what is this about?"}], 512))
        out[model] = {"classify": a, "rag": b}
        if a["ok"]:
            print(f"  OK   {model:<46} classify {a['s']:>5.2f}s "
                  f"out={a['out']} reasoning={a['reasoning']}"
                  f"{'  | rag ' + str(b['s']) + 's' if b['ok'] else '  | rag FAILED'}")
        else:
            print(f"  FAIL {model:<46} {a['err']}")
    (ROOT / "eval" / "bench_results.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
