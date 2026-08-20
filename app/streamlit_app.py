from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.bootstrap import build_stack, setup_logging  # noqa: E402
from src.config import get_config  # noqa: E402
from src.llm.router_llm import AllProvidersExhausted  # noqa: E402

st.set_page_config(page_title="Adaptive RAG", page_icon="🧭", layout="wide")

# Colours are set with alpha over the theme background so the same rules read
# correctly in both light and dark mode.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 1150px; }
      .hero h1 { margin-bottom: .1rem; font-size: 2.1rem; }
      .hero p  { color: rgba(128,128,128,.95); margin-top: 0; font-size: .95rem; }

      .pill {
        display:inline-block; padding:.28rem .8rem; border-radius:999px;
        font-weight:600; font-size:.82rem; letter-spacing:.02em;
      }
      .pill-naive   { background: rgba(46,160,67,.18);  color:#2ea043; border:1px solid rgba(46,160,67,.4); }
      .pill-agentic { background: rgba(137,87,229,.18); color:#a371f7; border:1px solid rgba(137,87,229,.45); }

      .subq {
        border-left: 3px solid rgba(137,87,229,.65);
        padding:.45rem .8rem; margin:.35rem 0;
        background: rgba(137,87,229,.07); border-radius:0 6px 6px 0;
        font-size:.92rem;
      }
      .subq b { color:#a371f7; margin-right:.4rem; }

      .src-meta { color: rgba(128,128,128,.95); font-size:.8rem; }
      div[data-testid="stMetricValue"] { font-size:1.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models and vector store…")
def get_stack():
    """Cached so the embedding model loads once per session, not per query."""
    setup_logging()
    return build_stack(get_config())


stack = get_stack()
cfg = stack.config

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Configuration")
    st.caption("Edit `config.yaml` and restart the app to change any of this.")

    st.metric("Indexed chunks", f"{stack.store.count():,}")

    st.markdown("**Retrieval**")
    st.code(stack.retrieval.describe(), language=None)
    enabled = [n for n, on in cfg.retrieval.components.items() if on]
    disabled = [n for n, on in cfg.retrieval.components.items() if not on]
    st.caption(f"on: {', '.join(enabled) or '—'}")
    st.caption(f"off: {', '.join(disabled) or '—'}")
    st.caption(f"top_k {cfg.retrieval.top_k} · parents by {cfg.chunking.parent.granularity}")

    st.divider()
    st.markdown("**Router**")
    if cfg.router.enabled:
        st.caption(f"enabled · paths: {', '.join(stack.router.available_paths)}")
    else:
        st.caption(f"disabled · everything → `{cfg.router.default_path}`")

    st.markdown("**Models by role**")
    for role, deps in cfg.llm.roles.items():
        st.caption(f"`{role}` → {deps[0].model}")
        if len(deps) > 1:
            st.caption(f"    ↳ {len(deps) - 1} fallback(s)")

    st.divider()
    st.caption(
        f"agentic caps: {cfg.agentic.max_sub_questions} hops · "
        f"{cfg.agentic.max_loops} retries/hop"
    )

# ---------------------------------------------------------------- header
st.markdown(
    """
    <div class="hero">
      <h1>🧭 Adaptive RAG</h1>
      <p>Each question is classified and sent down the cheapest path that can answer it —
         a single retrieval for simple lookups, a multi-hop agent for the rest.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if stack.store.count() == 0:
    st.warning(
        "Nothing indexed yet. Drop files in `data/` and run `python scripts/ingest.py`."
    )

with st.form("ask", clear_on_submit=False):
    query = st.text_input(
        "Ask a question",
        placeholder="Ask anything about the documents in data/…",
    )
    submitted = st.form_submit_button("Ask", type="primary")

# ---------------------------------------------------------------- run
if submitted and query.strip():
    st.divider()

    # Slots laid out first so each stage fills its own place as it arrives.
    route_slot = st.empty()
    subs_slot = st.container()
    steps_box = st.status("Working…", expanded=True)
    answer_slot = st.container()

    result = None
    started = time.time()

    try:
        for event in stack.stream(query):
            if event.kind == "route":
                route = event.route
                with route_slot.container():
                    css = "pill-agentic" if route.path == "agentic" else "pill-naive"
                    bits = [f'<span class="pill {css}">{route.path}</span>']
                    if route.complexity:
                        bits.append(
                            f'<span class="src-meta">&nbsp;classified '
                            f'<b>{route.complexity}</b>'
                            + (f" — {route.reason}" if route.reason else "")
                            + "</span>"
                        )
                    elif not route.classified:
                        bits.append(
                            f'<span class="src-meta">&nbsp;{route.reason or "router disabled"}</span>'
                        )
                    st.markdown("".join(bits), unsafe_allow_html=True)
                    if route.degraded:
                        st.info(
                            f"Classifier chose `{route.requested_path}`, which isn't "
                            f"available — ran `{route.path}` instead."
                        )

            elif event.kind == "sub_questions":
                with subs_slot:
                    st.markdown(
                        f"**Broken into {len(event.sub_questions)} sub-question"
                        f"{'s' if len(event.sub_questions) != 1 else ''}**"
                    )
                    for i, sub in enumerate(event.sub_questions, start=1):
                        st.markdown(
                            f'<div class="subq"><b>{i}.</b>{sub}</div>',
                            unsafe_allow_html=True,
                        )

            elif event.kind == "status":
                steps_box.write(event.message)

            elif event.kind == "result":
                result = event.result

    except AllProvidersExhausted as exc:
        steps_box.update(label="Stopped — providers unavailable", state="error")
        st.error(
            "Every configured LLM provider is rate limited or unavailable right now. "
            "Wait for the quota window to reset, or point a role at a different "
            "provider in `config.yaml`."
        )
        st.caption(str(exc))
        st.stop()

    elapsed = time.time() - started
    steps_box.update(label=f"Processing steps · {elapsed:.1f}s", state="complete", expanded=False)

    if result is None:
        st.error("The pipeline finished without producing an answer.")
        st.stop()

    with answer_slot:
        m = st.columns(4)
        m[0].metric("Path", result.path)
        m[1].metric("LLM calls", result.llm_calls or "—")
        m[2].metric("Sources", len(result.sources))
        m[3].metric("Time", f"{elapsed:.1f}s")

        if result.fell_back:
            st.warning(
                f"Primary provider was unavailable — answered with "
                f"`{result.provider}/{result.model}`."
            )
        if not result.grounded:
            st.warning(
                "The self-check judged this answer not fully grounded in the retrieved "
                "evidence, and the retry budget was exhausted. Treat it with caution."
            )

        st.markdown("### Answer")
        # Native bordered container, not a hand-rolled <div>: each st.markdown
        # call renders in its own DOM node, so a manual wrapper never closes
        # around the content.
        with st.container(border=True):
            st.markdown(result.answer)
        if result.model:
            st.caption(f"generated by {result.provider}/{result.model}")

        if result.truncated:
            st.warning(
                "This answer hit the length cap and was cut off. Raise "
                "`generation.max_answer_tokens` in `config.yaml` and restart."
            )

    st.markdown(f"### Sources ({len(result.sources)})")
    if not result.sources:
        st.caption("No sources retrieved.")
    for i, source in enumerate(result.sources, start=1):
        with st.expander(f"[{i}]  {source.label}", expanded=False):
            st.markdown(
                f'<span class="src-meta">score {source.best_score:.3f} · '
                f'{len(source.child_scores)} matching chunk(s) · '
                f'parent {source.parent_id[:12]}</span>',
                unsafe_allow_html=True,
            )
            st.text(source.text)

    with st.expander("Trace"):
        for line in result.trace:
            st.caption(line)
