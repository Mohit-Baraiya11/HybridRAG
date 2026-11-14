"""
STEP 2 of 2 - run in a SEPARATE venv with the pinned RAGAS stack (see requirements-ragas.txt).

Scores retrieval quality with RAGAS:
  - context_precision : are the retrieved chunks relevant, and are relevant ones ranked first?
  - context_recall    : is everything needed for the reference answer present in the chunks?
Both use the golden dataset's reference answers; no generated answers are needed.

    python eval_ragas.py --input retrieval_run.json --judge-model openai/gpt-oss-120b
    python eval_ragas.py --input retrieval_run.json --limit 8        # cheap trial run

Needs GROQ_API_KEY in the environment (or a .env file next to this script).
Each question costs ~(k + 1) judge calls, so a full 31-question run at k=5 is ~190 calls;
use --limit and --workers 1 if you hit Groq free-tier rate limits.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))              # the eval/ folder (or wherever this file lives)
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "eval" else HERE   # project root, so `tools.hybrid_rag_tool` imports work
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
from ragas.run_config import RunConfig


def build_judge(model_name: str):
    from langchain_groq import ChatGroq
    try:
        llm = ChatGroq(model=model_name, temperature=0, reasoning_effort="low")
    except Exception:
        llm = ChatGroq(model=model_name, temperature=0)
    return LangchainLLMWrapper(llm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(HERE, "retrieval_run.json"))
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N answerable questions")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", default=os.path.join(HERE, "ragas_results.json"))
    args = ap.parse_args()

    run = json.load(open(args.input))
    rows = [r for r in run["rows"] if r["answerable"]]
    empty = [r["id"] for r in rows if not r["retrieved_contexts"]]
    if len(empty) == len(rows):
        sys.exit("STOP: every question has zero retrieved contexts, so RAGAS scores would be meaningless. "
                 "Fix retrieval/ingestion and re-run collect_retrievals.py.")
    if empty:
        print(f"WARNING: {len(empty)} questions have no retrieved contexts and are skipped: {empty}")
        rows = [r for r in rows if r["retrieved_contexts"]]
    if args.limit:
        rows = rows[: args.limit]

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not set.")
    judge = build_judge(args.judge_model)

    print(f"Scoring {len(samples)} questions with RAGAS (judge: {args.judge_model}) ...")
    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=judge,
        run_config=RunConfig(max_workers=args.workers, timeout=args.timeout, max_retries=6),
        raise_exceptions=False,
        show_progress=True,
    )
    df = result.to_pandas()
    p_col = [c for c in df.columns if "precision" in c][0]
    r_col = [c for c in df.columns if "recall" in c][0]

    per_q = []
    for i, r in enumerate(rows):
        per_q.append({
            "id": r["id"], "type": r["type"], "question": r["question"],
            "context_precision": None if df[p_col].isna()[i] else round(float(df[p_col][i]), 3),
            "context_recall": None if df[r_col].isna()[i] else round(float(df[r_col][i]), 3),
            "anchor_rank": r["anchor_rank"],
        })

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    overall = {"n": len(per_q),
               "context_precision": mean([q["context_precision"] for q in per_q]),
               "context_recall": mean([q["context_recall"] for q in per_q]),
               "failed_scores": sum(1 for q in per_q if q["context_precision"] is None or q["context_recall"] is None)}
    by_type = defaultdict(list)
    for q in per_q:
        by_type[q["type"]].append(q)

    print("\n=== RAGAS retrieval scores ===")
    print(json.dumps(overall))
    for t, qs in by_type.items():
        print(f"  {t:12} n={len(qs):2}  precision={mean([q['context_precision'] for q in qs])}  "
              f"recall={mean([q['context_recall'] for q in qs])}")

    worst = sorted([q for q in per_q if q["context_recall"] is not None],
                   key=lambda q: (q["context_recall"], q["context_precision"] or 0))[:8]
    print("\nLowest-recall questions (look at these first):")
    for q in worst:
        print(f"  {q['id']} [{q['type']}] recall={q['context_recall']} precision={q['context_precision']} "
              f"anchor_rank={q['anchor_rank']} | {q['question'][:70]}")

    json.dump({"config": run["config"], "judge": args.judge_model, "overall": overall,
               "by_type": {t: {"n": len(v), "context_precision": mean([x['context_precision'] for x in v]),
                               "context_recall": mean([x['context_recall'] for x in v])} for t, v in by_type.items()},
               "per_question": per_q}, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()