"""
retrieval_eval.py - ONE file, runs in your normal project venv, no RAGAS needed.

For every question in golden_dataset.json it calls YOUR hybrid_rag.retrive()
(embed -> Pinecone hybrid query -> rerank -> parent chunks), then checks whether
the evidence text for that question is inside the chunks that came back.

    python retrieval_eval.py --user-id <uuid> --session-id <uuid> --module agents.hybrid_rag
    python retrieval_eval.py ... -v                 # also print the retrieved chunk previews
    python retrieval_eval.py ... --no-rerank        # compare without the reranker
    python retrieval_eval.py ... --alpha 1.0        # dense only   (0.0 = BM25/sparse only)

The step timings (embed / pinecone / rerank) are printed by retrive() itself.
"""
import argparse, importlib, json, os, re, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))              # the eval/ folder (or wherever this file lives)
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "eval" else HERE   # project root, so `tools.hybrid_rag_tool` imports work


def key(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def check(item, contexts):
    """Compare retrieved contexts with the golden evidence. Returns (rank, anchors_found, anchors_total)."""
    keyed = [key(c) for c in contexts]
    per_anchor = []
    for a in item["anchors"]:
        r = [i + 1 for i, c in enumerate(keyed) if key(a) in c]
        per_anchor.append(min(r) if r else None)
    found = [r for r in per_anchor if r]
    if item.get("require_all_anchors"):
        rank = max(per_anchor) if len(found) == len(per_anchor) else None
    else:
        rank = min(found) if found else None
    return rank, len(found), len(per_anchor)


def namespace_vector_count(index, session_id):
    stats = index.describe_index_stats()
    ns = getattr(stats, "namespaces", None) or stats["namespaces"]
    entry = ns.get(session_id)
    if entry is None:
        return 0
    return getattr(entry, "vector_count", None) or entry["vector_count"]


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
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "retrieval_eval_results.json"))
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, ROOT)
    mod = importlib.import_module(args.module)
    mod.USE_RERANK = not args.no_rerank

    # ---- preflight: are the vectors actually there? ----
    try:
        n_vec = namespace_vector_count(mod.get_index(), args.session_id)
        print(f"Pinecone vectors in namespace {args.session_id}: {n_vec}")
        if n_vec == 0:
            sys.exit("STOP: 0 vectors for this session. Ingestion did not finish (embedding/upsert step failed) "
                     "or this is the wrong session id. Re-upload the PDF in a NEW chat and check the server log "
                     "for 'Background ingestion failed'.")
    except SystemExit:
        raise
    except Exception as e:
        print(f"(preflight vector-count check skipped: {e})")

    bm25_enc = mod.get_bm25(f"{args.user_id}_{args.session_id}")
    parents = mod.get_parents(args.user_id, args.session_id)
    if bm25_enc is None or not parents:
        sys.exit("No BM25/parents for this user+session id.")
    print(f"Loaded {len(parents)} parent chunks | k={args.k} alpha={args.alpha} rerank={not args.no_rerank}\n")

    data = json.load(open(args.dataset))
    rows = []
    for item in [d for d in data if d["answerable"]]:
        print(f"[{item['id']}] ({item['type']}) {item['question']}")
        t0 = time.time()
        ctxs = mod.retrive(item["question"], bm25_enc, parents, top_k=args.k,
                           alpha=args.alpha, session_id=args.session_id)
        ms = round((time.time() - t0) * 1000)
        rank, nfound, ntotal = check(item, ctxs)
        status = f"HIT at rank {rank}" if rank else f"MISS ({nfound}/{ntotal} evidence found)"
        print(f"   retrieved {len(ctxs)} chunks in {ms}ms -> {status} | expected PDF pages {item['relevant_pdf_pages']}")
        if args.verbose or not rank:
            for i, c in enumerate(ctxs, 1):
                mark = "✓" if any(key(a) in key(c) for a in item["anchors"]) else " "
                print(f"     {mark} #{i}: {c[:110].replace(chr(10), ' ')}")
        rows.append({"id": item["id"], "type": item["type"], "question": item["question"],
                     "rank": rank, "anchors_found": nfound, "anchors_total": ntotal, "ms": ms,
                     "n_contexts": len(ctxs), "expected_pdf_pages": item["relevant_pdf_pages"]})
        time.sleep(args.sleep)

    def stats(rs):
        n = len(rs)
        h = lambda j: round(sum(1 for r in rs if r["rank"] and r["rank"] <= j) / n, 2)
        return {"n": n, "hit@1": h(1), "hit@3": h(3), f"hit@{args.k}": h(args.k),
                "mrr": round(sum(1 / r["rank"] for r in rs if r["rank"]) / n, 2),
                "avg_ms": round(sum(r["ms"] for r in rs) / n)}

    print("\n=========== SUMMARY ===========")
    print("overall:", stats(rows))
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    for t, rs in by_type.items():
        s = stats(rs)
        print(f"  {t:12} n={s['n']:2}  hit@1={s['hit@1']:.2f}  hit@{args.k}={s[f'hit@{args.k}']:.2f}  mrr={s['mrr']:.2f}")
    misses = [r for r in rows if not r["rank"]]
    print(f"\nmissed {len(misses)}/{len(rows)}:")
    for r in misses:
        print(f"  {r['id']} [{r['type']}] PDF pages {r['expected_pdf_pages']} | {r['question'][:70]}")

    json.dump({"args": vars(args), "overall": stats(rows), "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()