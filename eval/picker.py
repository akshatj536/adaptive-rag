"""Question picker for building the evaluation set.

Browse HotpotQA with filters, mark each question as a naive or agentic
candidate, and export the selection. The export carries each question's full
context paragraphs, so the eval corpus can be built later without touching the
dataset again.

    .venv/bin/streamlit run eval/picker.py --server.port 8502
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import streamlit as st
from datasets import load_dataset

OUT_PATH = Path(__file__).parent / "selected_questions.jsonl"

st.set_page_config(page_title="Eval question picker", page_icon="✓", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1250px; }
      .qcard { border-left: 3px solid rgba(128,128,128,.35); padding: .1rem 0 .1rem .8rem; }
      .qcard.naive   { border-left-color: #2ea043; }
      .qcard.agentic { border-left-color: #a371f7; }
      .meta { color: rgba(128,128,128,.95); font-size: .8rem; }
      .ans { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading HotpotQA…")
def get_dataset():
    return load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")


def _norm(title: str) -> str:
    """Strip the disambiguation suffix: 'Tron (franchise)' -> 'tron'."""
    return re.sub(r"\(.*?\)", "", title).strip().lower()


INDEX_PATH = Path(__file__).parent / "hotpot_index.json"


@st.cache_data(show_spinner="Indexing questions (one-off, ~15s)…")
def build_index() -> list[dict]:
    """Lightweight searchable index, cached to disk so the cost is paid once.

    The `list(...)` calls matter: indexing a datasets Column row by row is lazy
    and re-materializes each access, which turned this loop into a five-minute
    hang. Forcing the columns into Python lists first takes it to about a second.
    """
    if INDEX_PATH.exists():
        cached = json.loads(INDEX_PATH.read_text())
        if cached and "id" in cached[0]:
            return cached  # otherwise fall through and rebuild

    ds = get_dataset()
    ids = list(ds["id"])
    questions = list(ds["question"])
    answers = list(ds["answer"])
    levels = list(ds["level"])
    types = list(ds["type"])
    facts = list(ds["supporting_facts"])

    index = []
    for i in range(len(questions)):
        gold = list(dict.fromkeys(facts[i]["title"]))
        q_lower = questions[i].lower()
        named = sum(1 for t in gold if _norm(t) and _norm(t) in q_lower)
        index.append(
            {
                # id is the stable identifier and is what selections are keyed
                # by; row is only how we fetch the heavy context column.
                "id": ids[i],
                "row": i,
                "question": questions[i],
                "answer": answers[i],
                "level": levels[i],
                "type": types[i],
                "n_gold": len(gold),
                # Fraction of gold titles the question names outright. The real
                # difficulty signal: comparisons name both (1.0), genuine
                # multi-hop bridges name none (0.0).
                "named": named / max(1, len(gold)),
            }
        )

    INDEX_PATH.write_text(json.dumps(index))
    return index


def full_record(row: int, expected_path: str) -> dict:
    """Everything needed to rebuild the corpus and score the answer later."""
    r = get_dataset()[row]
    return {
        "id": r["id"],
        "question": r["question"],
        "answer": r["answer"],
        "type": r["type"],
        "level": r["level"],
        "expected_path": expected_path,
        "supporting_facts": {
            "title": list(r["supporting_facts"]["title"]),
            "sent_id": [int(s) for s in r["supporting_facts"]["sent_id"]],
        },
        "context": {
            "title": list(r["context"]["title"]),
            "sentences": [list(s) for s in r["context"]["sentences"]],
        },
    }


# ---------------------------------------------------------------- state
if "picks" not in st.session_state:
    st.session_state.picks = {}          # row -> record
if "seed" not in st.session_state:
    st.session_state.seed = 7

index = build_index()

# ---------------------------------------------------------------- sidebar
PRESETS = {
    "Agentic candidates — bridge, easy/medium": dict(
        types=["bridge"], levels=["easy", "medium"], answer_shape="any", max_named=1.0
    ),
    "Naive candidates — comparison, proper-noun answer": dict(
        types=["comparison"], levels=["easy", "medium", "hard"],
        answer_shape="starts uppercase, not yes/no", max_named=1.0,
    ),
    "Hard multi-hop — bridge, nothing named": dict(
        types=["bridge"], levels=["hard"], answer_shape="any", max_named=0.0
    ),
    "Custom": None,
}

with st.sidebar:
    st.markdown("### Filters")
    preset_name = st.selectbox("Preset", list(PRESETS), index=0)
    preset = PRESETS[preset_name]

    if preset:
        types = preset["types"]
        levels = preset["levels"]
        answer_shape = preset["answer_shape"]
        max_named = preset["max_named"]
        st.caption(f"type: {', '.join(types)}")
        st.caption(f"level: {', '.join(levels)}")
        st.caption(f"answer: {answer_shape}")
        st.caption(f"max gold titles named: {max_named:.0%}")
    else:
        types = st.multiselect("Type", ["bridge", "comparison"], default=["bridge"])
        levels = st.multiselect("Level", ["easy", "medium", "hard"], default=["easy", "medium"])
        answer_shape = st.selectbox(
            "Answer shape", ["any", "starts uppercase, not yes/no", "exclude yes/no"]
        )
        max_named = st.slider(
            "Max fraction of gold titles named in the question", 0.0, 1.0, 1.0, 0.5,
            help="0.0 keeps only questions naming none of their gold titles, "
                 "which is the genuine multi-hop case.",
        )

    keyword = st.text_input("Keyword in question", placeholder="optional")
    n_show = st.slider("How many to show", 5, 50, 15, 5)

    st.divider()
    if st.button("Resample", use_container_width=True):
        st.session_state.seed = random.randint(1, 10**6)
        st.rerun()
    st.caption(f"seed {st.session_state.seed}")

    st.divider()
    # Reserved now, filled after the rows are rendered. The row loop is what
    # mutates `picks`, so populating this here would always show the previous
    # run's state and the Save button would appear one interaction late.
    selection_slot = st.container()


# ---------------------------------------------------------------- filtering
def keep(item: dict) -> bool:
    if item["type"] not in types or item["level"] not in levels:
        return False
    if item["named"] > max_named:
        return False
    a = item["answer"]
    if answer_shape == "starts uppercase, not yes/no":
        if a.lower() in ("yes", "no") or not a[:1].isupper():
            return False
    elif answer_shape == "exclude yes/no" and a.lower() in ("yes", "no"):
        return False
    if keyword and keyword.lower() not in item["question"].lower():
        return False
    return True


pool = [it for it in index if keep(it)]
rng = random.Random(st.session_state.seed)
sample = rng.sample(pool, min(n_show, len(pool))) if pool else []

st.title("Evaluation question picker")
st.caption(
    f"{len(pool):,} questions match these filters out of {len(index):,}. "
    f"Showing {len(sample)}. Selections persist when you change filters or resample."
)

if not pool:
    st.warning("No questions match. Loosen the filters.")
    st.stop()

# ---------------------------------------------------------------- rows
OPTIONS = ["skip", "naive", "agentic"]

for item in sample:
    qid, row = item["id"], item["row"]
    current = st.session_state.picks.get(qid, {}).get("expected_path", "skip")
    css = current if current in ("naive", "agentic") else ""

    left, right = st.columns([5, 1.5])
    with left:
        st.markdown(
            f'<div class="qcard {css}">{item["question"]}<br>'
            f'<span class="ans">→ {item["answer"]}</span><br>'
            f'<span class="meta">{item["type"]} · {item["level"]} · '
            f'{item["n_gold"]} gold passage(s) · '
            f'{item["named"]:.0%} of them named in the question</span></div>',
            unsafe_allow_html=True,
        )
    with right:
        choice = st.radio(
            "assign", OPTIONS, index=OPTIONS.index(current),
            key=f"pick_{qid}", horizontal=False, label_visibility="collapsed",
        )

    if choice == "skip":
        st.session_state.picks.pop(qid, None)
    elif st.session_state.picks.get(qid, {}).get("expected_path") != choice:
        st.session_state.picks[qid] = full_record(row, choice)

    with st.expander("context paragraphs"):
        r = get_dataset()[row]
        gold = set(r["supporting_facts"]["title"])
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            tag = "**GOLD**" if title in gold else "distractor"
            st.markdown(f"{tag} · **{title}** — {' '.join(sents)[:400]}…")

    st.divider()


# ------------------------------------------------- selection panel (sidebar)
picks = st.session_state.picks
with selection_slot:
    st.markdown("### Selected")
    c1, c2 = st.columns(2)
    c1.metric("naive", sum(1 for p in picks.values() if p["expected_path"] == "naive"))
    c2.metric("agentic", sum(1 for p in picks.values() if p["expected_path"] == "agentic"))

    if picks:
        payload = "\n".join(json.dumps(p, ensure_ascii=False) for p in picks.values())
        if st.button("Save to eval/selected_questions.jsonl", type="primary",
                     use_container_width=True):
            OUT_PATH.write_text(payload + "\n", encoding="utf-8")
            st.success(f"Wrote {len(picks)} question(s)")
        st.download_button("Download .jsonl", payload + "\n",
                           file_name="selected_questions.jsonl",
                           use_container_width=True)
        if st.button("Clear selection", use_container_width=True):
            # The radio widgets own their state under `pick_*` keys. Clearing
            # only `picks` is not enough: on the next run the row loop reads
            # those widgets back and rebuilds the selection.
            for key in [k for k in st.session_state if k.startswith("pick_")]:
                del st.session_state[key]
            st.session_state.picks = {}
            st.rerun()
    else:
        st.caption("Assign a question to naive or agentic to enable saving.")
