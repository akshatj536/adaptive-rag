#!/usr/bin/env python3
"""Run both paths over the selected questions and cache every result.

Results are appended to eval/results.jsonl keyed by (question id, path). Re-runs
skip whatever is already recorded, so a run interrupted by a rate-limit cap can
simply be restarted. Nothing is ever paid for twice.

The adaptive path is never executed: it is derived later by taking the
classifier's label per question and selecting that path's recorded answer.

    python eval/runner.py --paths naive              # cheap, 1 call per question
    python eval/runner.py --paths agentic --limit 10
    python eval/runner.py --paths route              # classifier label only
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bootstrap import build_stack, setup_logging   # noqa: E402
from src.config import get_config                      # noqa: E402
from src.llm.base import LLMError                      # noqa: E402

EVAL_DIR = ROOT / "eval"
QUESTIONS = EVAL_DIR / "selected_questions.jsonl"
RESULTS = EVAL_DIR / "results.jsonl"
TITLE_MAP = EVAL_DIR / "corpus_titles.json"
COLLECTION = "eval_hotpot"


def eval_config():
    cfg = get_config()
    return dataclasses.replace(
        cfg, vectorstore=dataclasses.replace(cfg.vectorstore, collection=COLLECTION)
    )


def load_done() -> set[tuple[str, str]]:
    if not RESULTS.exists():
        return set()
    done = set()
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["id"], r["path"]))
    return done


def append(record: dict) -> None:
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["naive"],
                    choices=["naive", "agentic", "route"])
    ap.add_argument("--limit", type=int, default=None, help="max questions per path")
    ap.add_argument("--only", choices=["naive", "agentic"],
                    help="restrict to questions you tagged for this path")
    args = ap.parse_args()

    setup_logging()
    cfg = eval_config()
    stack = build_stack(cfg)
    titles = json.loads(TITLE_MAP.read_text())

    questions = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    if args.only:
        questions = [q for q in questions if q["expected_path"] == args.only]
    done = load_done()
    print(f"{len(questions)} question(s) | {len(done)} result(s) already cached | "
          f"store has {stack.store.count()} chunks")

    for path in args.paths:
        todo = [q for q in questions if (q["id"], path) not in done]
        if args.limit:
            todo = todo[: args.limit]
        print(f"\n=== {path}: {len(todo)} to run ===")

        for i, q in enumerate(todo, 1):
            started = time.time()
            record = {"id": q["id"], "path": path, "question": q["question"],
                      "gold_answer": q["answer"], "expected_path": q["expected_path"],
                      "type": q["type"], "level": q["level"]}
            try:
                if path == "route":
                    decision = stack.router.decide(q["question"])
                    record |= {"routed_to": decision.path,
                               "complexity": decision.complexity,
                               "reason": decision.reason,
                               "classified": decision.classified,
                               "llm_calls": 1}
                else:
                    res = stack.pipelines[path].run(q["question"])
                    record |= {
                        "answer": res.answer,
                        "llm_calls": res.llm_calls,
                        "sub_questions": res.sub_questions,
                        "grounded": res.grounded,
                        "retrieved_titles": [titles.get(s.source, s.source) for s in res.sources],
                        "n_sources": len(res.sources),
                        "provider": res.provider,
                        "model": res.model,
                        "fell_back": res.fell_back,
                    }
            except LLMError as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 - never lose the rest of the run
                record["error"] = f"{type(exc).__name__}: {exc}"

            record["elapsed_s"] = round(time.time() - started, 2)
            append(record)

            flag = "ERR " if record.get("error") else "    "
            print(f"  {flag}[{i}/{len(todo)}] {record['elapsed_s']:6.1f}s "
                  f"calls={record.get('llm_calls', 0):<3} {q['question'][:58]}")
            if record.get("error"):
                print(f"        {record['error'][:110]}")
                if "Exhausted" in record["error"]:
                    print("\nProviders exhausted. Re-run this command later; "
                          "cached results are kept.")
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
