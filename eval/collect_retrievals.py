"""
STEP 1 of 2 - run in YOUR PROJECT venv (needs pinecone/supabase, NOT ragas).

Runs your retriever over golden_dataset.json and saves exactly what the
synthesizer would receive, so RAGAS can score it in a separate venv (step 2).

    python collect_retrievals.py --user-id <uuid> --session-id <uuid> \
        --module agents.hybrid_rag --k 5

Options:
    --no-rerank     collect with USE_RERANK=False  (to compare against reranked)
    --alpha 0.5     hybrid weight (1.0 = dense only, 0.0 = sparse only)
    --out retrieval_run.json
"""
import argparse, importlib, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))              # the eval/ folder (or wherever this file lives)
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "eval" else HERE   # project root, so `tools.hybrid_rag_tool` imports work


def key(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def anchor_rank(item, contexts):
    """Deterministic sanity check: rank of first context containing an evidence anchor."""
    keyed = [key(c) for c in contexts]
    ranks = []
    for a in item["anchors"]:
        r = [i + 1 for i, c in enumerate(keyed) if key(a) in c]
        ranks.append(min(r) if r else None)
    if not ranks:
        return None
    if item.get("require_all_anchors"):
        return None if any(r is None for r in ranks) else max(ranks)
    found = [r for r in ranks if r]
    return min(found) if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="agents.hybrid_rag")
    ap.add_argument("--dataset", default=os.path.join(HERE, "golden_dataset.json"))
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--out", default=os.path.join(HERE, "retrieval_run.json"))
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, ROOT)
    mod = importlib.import_module(args.module)
    mod.USE_RERANK = not args.no_rerank

    data = json.load(open(args.dataset))
    bm25_enc = mod.get_bm25(f"{args.user_id}_{args.session_id}")
    parents = mod.get_parents(args.user_id, args.session_id)
    if bm25_enc is None or not parents:
        sys.exit("No BM25/parents for this user+session. Upload the PDF in that session first.")

    rows = []
    for item in data:
        t0 = time.time()
        ctxs = mod.retrive(item["question"], bm25_enc, parents, top_k=args.k,
                           alpha=args.alpha, session_id=args.session_id)
        rows.append({
            "id": item["id"], "type": item["type"], "answerable": item["answerable"],
            "question": item["question"], "reference": item["ground_truth"],
            "retrieved_contexts": ctxs,
            "anchor_rank": anchor_rank(item, ctxs) if item["answerable"] else None,
            "retrieval_ms": round((time.time() - t0) * 1000),
        })
        time.sleep(args.sleep)
        print(f"{item['id']}: {len(ctxs)} contexts, anchor_rank={rows[-1]['anchor_rank']}")
        if len(rows) == 3 and all(len(r["retrieved_contexts"]) == 0 for r in rows):
            sys.exit(
                "\nSTOP: the first 3 queries returned 0 contexts. BM25/parents loaded but Pinecone has no vectors "
                "for this session (ingestion probably failed at the embedding/upsert step, or the vectors are in a "
                "different namespace). Run idx.describe_index_stats() and check that your session id appears under "
                "'namespaces' with a vector count, then re-ingest. Nothing was saved."
            )

    json.dump({"config": {"k": args.k, "alpha": args.alpha, "rerank": not args.no_rerank},
               "rows": rows}, open(args.out, "w"), indent=2, ensure_ascii=False)
    ans = [r for r in rows if r["answerable"]]
    hit5 = sum(1 for r in ans if r["anchor_rank"]) / len(ans)
    print(f"\nSaved {args.out}  |  anchor Hit@{args.k} = {hit5:.2f}  (quick sanity check; RAGAS scoring is step 2)")


if __name__ == "__main__":
    main()