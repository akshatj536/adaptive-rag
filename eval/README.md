# Evaluation

Work in progress. This directory builds the question set for evaluating the
router; the harness that runs and scores it does not exist yet.

## What is being evaluated

Not "is the RAG accurate". The project's claim is that **routing preserves
answer quality while cutting cost**, so the evaluation is a routing evaluation:

- Run every question through **both** paths (naive and agentic).
- The adaptive result is then *derived*, not run: for each question, take the
  classifier's label and select that path's already-computed answer. Running
  adaptive separately would be paying twice for the same numbers.
- Compare against an **oracle** router, which picks the correct path with
  hindsight. That is the ceiling; the gap between adaptive and oracle is what
  the classifier is leaving on the table.

The decisive quantity is the fraction of questions where **naive fails and
agentic succeeds**. That cell is the only place routing can demonstrate value,
and it sets the sample size: aim for at least 30 questions in it, so
`N ≈ 30 / decisive_rate`.

## Dataset: HotpotQA (distractor, train)

Chosen because it ships three things this evaluation needs:

- multi-hop questions, so the agentic path has a reason to exist
- `supporting_facts` at sentence level, which gives retrieval ground truth
- `level` and `type` labels for stratification

The train split is used rather than dev because dev-distractor is uniformly
hard, and the experiment needs easy questions too. Nothing is being trained.

### What the data actually shows

Measured over a 20k-row slice, the fraction of questions that name **all** of
their gold passage titles in the question text:

| level | type | all named | none named |
|---|---|---|---|
| easy | comparison | 100% | 0% |
| medium | comparison | 100% | 0% |
| hard | comparison | 100% | 0% |
| easy | bridge | 25% | 26% |
| medium | bridge | 6% | 29% |
| hard | bridge | 8% | 32% |

**`level` barely predicts retrieval difficulty; `type` does.** Comparison
questions name both entities by construction ("which came first, X or Y"), so a
single retrieval has a real chance. Bridge questions name all their gold titles
only 6-25% of the time, and about 30% name none at all, which is what forces a
second hop.

So difficulty is stratified by `type` plus the entity-naming feature, not by the
dataset's own `level` field.

One caveat when picking: roughly 32% of comparison answers are yes/no, and
others are generic ("band", "university"). Those are guessable without
retrieval, so both paths score them correct and they contribute nothing to the
decisive cell.

## Picker

```bash
.venv/bin/streamlit run eval/picker.py --server.port 8502
```

Browse with filters, mark each question `skip` / `naive` / `agentic`, and export.

Presets encode the strata above:

| Preset | Filters |
|---|---|
| Agentic candidates | bridge, easy + medium |
| Naive candidates | comparison, uppercase answer, no yes/no |
| Hard multi-hop | bridge, hard, names none of its gold titles |
| Custom | everything manual |

Selections persist across filter changes and resampling, and are keyed by
question `id` rather than row index.

**Output:** `eval/selected_questions.jsonl`, one record per question carrying
the question, gold answer, `supporting_facts`, the expected path, and **all ten
context paragraphs**. Embedding the context makes the file self-contained, so
the eval corpus can be rebuilt later without touching the dataset again.

On first run the picker builds `eval/hotpot_index.json` (~21 MB, gitignored) and
reuses it afterwards. Building it takes about 20 seconds; loading it takes 0.3.

## Corpus construction (next step)

Write **one paragraph per file**. HotpotQA paragraphs are ~486 characters
median, so one paragraph becomes one page becomes one parent in the chunker.
Parent granularity then matches the dataset's annotation granularity, and
retrieval ground truth becomes an exact check ("did we retrieve a gold-titled
paragraph?") instead of a fuzzy span-overlap heuristic.

Index into a **separate collection** (`vectorstore.collection: eval_hotpot`) so
the working corpus is left alone.

## Still to build

- `runner.py` — run both paths per question, cache to JSONL, resume on failure
  (free-tier runs will hit caps mid-way)
- `metrics.py` — exact match / F1 first, LLM judge only where EM fails but the
  answer may still be right; recall@k; call counts
- `report.py` — the naive x agentic matrix, router confusion matrix, and a
  cost/quality plot with naive, agentic, adaptive, oracle, and a random router
  at the same agentic rate as a baseline
