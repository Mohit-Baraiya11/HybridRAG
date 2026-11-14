"""
fix_parent_store.py - removes orphaned rows from the Supabase parent_store table.

Failed / repeated ingestions each save a full set of parent chunks under a NEW
document id, but Pinecone only keeps vectors for one of them. This script:
  1. asks Pinecone which document id(s) are actually live in the session's namespace
  2. counts parent_store rows for the session: live vs orphans
  3. deletes the orphans (only when you pass --yes; the default is a dry run)

Run from the project root:
    uv run python eval/fix_parent_store.py --session-id <SESSION_ID>          # dry run, changes nothing
    uv run python eval/fix_parent_store.py --session-id <SESSION_ID> --yes    # actually deletes
"""
import argparse, importlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "eval" else HERE


def fetch_all_ids(supabase, session_id, page=1000):
    ids, start = [], 0
    while True:
        res = (supabase.table("parent_store").select("id")
               .eq("session_id", session_id).order("id")
               .range(start, start + page - 1).execute())
        ids += [r["id"] for r in res.data]
        if len(res.data) < page:
            return ids
        start += page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="tools.hybrid_rag_tool")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--yes", action="store_true", help="really delete (default is a dry run)")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, ROOT)
    mod = importlib.import_module(args.module)

    # 1) which document id(s) are live in Pinecone for this session?
    res = mod.get_index().query(vector=[0.0] * 1024, top_k=1000, include_metadata=True,
                                namespace=args.session_id)
    doc_ids = sorted({m["metadata"]["document_id"] for m in res["matches"]})
    if not doc_ids:
        sys.exit("STOP: Pinecone has no vectors for this session, so nothing can be called 'live'. Nothing deleted.")
    print(f"Live document id(s) in Pinecone: {doc_ids}")
    if len(doc_ids) > 1:
        print("  (more than one document in this session - all of them are treated as live)")

    # 2) count live vs orphan parent rows
    all_ids = fetch_all_ids(mod.supabase, args.session_id)
    live = [i for i in all_ids if any(i.startswith(d + "_p") for d in doc_ids)]
    orphans = [i for i in all_ids if i not in set(live)]
    print(f"parent_store rows for session: {len(all_ids)}  ->  live: {len(live)}  |  orphans: {len(orphans)}")

    if not live:
        sys.exit("STOP: no live rows matched the Pinecone document id. Not deleting anything.")
    if not orphans:
        print("Nothing to clean up.")
        return
    if not args.yes:
        print("\nDRY RUN - nothing was deleted. Re-run with --yes to delete the orphans.")
        return

    # 3) delete orphans in batches
    for i in range(0, len(orphans), 200):
        mod.supabase.table("parent_store").delete().in_("id", orphans[i:i + 200]).execute()
    remaining = len(fetch_all_ids(mod.supabase, args.session_id))
    print(f"Deleted {len(orphans)} orphan rows. parent_store rows now: {remaining}")


if __name__ == "__main__":
    main()