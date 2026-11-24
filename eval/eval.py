"""One-file evaluation tools for HybridRAG.

Examples (run from the project root):
  uv run python eval/eval.py retrieval --user-id ID --session-id ID
  python eval/eval.py ragas --input /tmp/hybridrag-retrieval-run.json
  uv run python eval/eval.py cleanup --session-id ID              # dry run
  uv run python eval/eval.py cleanup --session-id ID --yes        # delete orphans

RAGAS uses an incompatible LangChain dependency set, so run it in a separate venv:
  python3 -m venv .venv-ragas
  source .venv-ragas/bin/activate
  pip install 'ragas==0.3.9' 'langchain<1.0' 'langchain-core<1.0' \
    'langchain-community<0.4' 'langchain-openai<1.0' 'langchain-groq<1.0' pandas python-dotenv
  python eval/eval.py ragas --input /tmp/hybridrag-retrieval-run.json

Retrieval runs save their data to /tmp by default. Pass --out to choose another path.
"""
import argparse
import importlib
import json
import os
import re
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MODULE = "tools.hybrid_rag_tool"
DEFAULT_RUN = "/tmp/hybridrag-retrieval-run.json"

# Compact form of the golden dataset; keeping it here makes this the only eval file.
DATASET = [
    {"id":"q01","type":"factual","question":"How do you view the source code of a function in IPython?","ground_truth":"Use the double question mark (??) after the object name, e.g. `square??`. A single ? shows the docstring; ?? also shows the source code.","answerable":True,"anchors":["shortcut to the source code with the double question mark"],"relevant_pdf_pages":[23]},
    {"id":"q02","type":"paraphrase","question":"How can I make Python error tracebacks less verbose in IPython?","ground_truth":"Use the %xmode magic function. It takes one argument, the mode: Plain, Context (default) or Verbose. Plain is more compact and gives less information.","answerable":True,"anchors":["three possibilities: Plain, Context, and Verbose"],"relevant_pdf_pages":[39]},
    {"id":"q03","type":"conceptual","question":"Why can %timeit give a misleading result when timing a list sort, and what should be used instead?","ground_truth":"%timeit repeats the operation, and sorting an already-sorted list is much faster, so repetition skews the result. %time is the better choice for this case (and for longer-running commands).","answerable":True,"anchors":["so the repetition will skew the result"],"relevant_pdf_pages":[44]},
    {"id":"q04","type":"keyword","question":"Which package has to be installed to use the %lprun line-by-line profiler?","ground_truth":"The line_profiler package, installed with pip (pip install line_profiler). It is not built into Python or IPython.","answerable":True,"anchors":["there is a line_profiler package available for installation"],"relevant_pdf_pages":[46]},
    {"id":"q05","type":"factual","question":"What are the rules of broadcasting in NumPy?","ground_truth":"Rule 1: if the arrays differ in number of dimensions, the smaller shape is padded with ones on its left side. Rule 2: a dimension of size 1 is stretched to match the other array. Rule 3: if sizes still disagree in any dimension and neither is 1, an error is raised.","answerable":True,"anchors":["shape of the one with fewer dimensions is padded with ones on its leading (left) side"],"relevant_pdf_pages":[83]},
    {"id":"q06","type":"factual","question":"What does np.argsort return?","ground_truth":"The indices of the sorted elements: the first value is the index of the smallest element, the second the index of the second smallest, and so on. These can be used with fancy indexing to build the sorted array.","answerable":True,"anchors":["The first element of this result gives the index of the smallest element"],"relevant_pdf_pages":[105]},
    {"id":"q07","type":"keyword","question":"How can I increment array positions given repeated indices, since x[i] += 1 does not accumulate?","ground_truth":"Use the at() method of ufuncs: np.add.at(x, i, 1). It applies the operator in place at the specified indices.","answerable":True,"anchors":["does an in-place application of the given operator at the specified indices"],"relevant_pdf_pages":[101]},
    {"id":"q08","type":"factual","question":"What is the difference between loc and iloc in Pandas?","ground_truth":"loc indexes and slices using the explicit index; iloc always uses the implicit Python-style (positional) index.","answerable":True,"anchors":["always references the explicit index"],"relevant_pdf_pages":[128]},
    {"id":"q09","type":"factual","question":"What is a many-to-one join in pd.merge?","ground_truth":"A join where one of the two key columns contains duplicate entries; the resulting DataFrame preserves those duplicates as appropriate.","answerable":True,"anchors":["one of the two key columns contains duplicate entries"],"relevant_pdf_pages":[166]},
    {"id":"q10","type":"keyword","question":"How do I deal with overlapping column names when merging two DataFrames?","ground_truth":"Use the suffixes keyword of pd.merge, e.g. suffixes=['_L', '_R'], to disambiguate the conflicting column names.","answerable":True,"anchors":["your two input DataFrames have conflicting column names","These suffixes work in any of the possible join patterns"],"relevant_pdf_pages":[171,172]},
    {"id":"q11","type":"conceptual","question":"What does split-apply-combine mean in a Pandas groupby operation?","ground_truth":"A term from Hadley Wickham: split the DataFrame by a key, apply a function (aggregate, transform or filter) within each group, then combine the results into an output.","answerable":True,"anchors":["The split step involves breaking up and grouping a DataFrame"],"relevant_pdf_pages":[179]},
    {"id":"q12","type":"conceptual","question":"When is it worth using eval() or query() in Pandas?","ground_truth":"Compound expressions create temporary arrays; when those are large relative to available memory, eval()/query() avoid them. eval() can also be faster for large arrays, but not for small ones.","answerable":True,"anchors":["every compound expression involving NumPy arrays or Pandas DataFrames will result in implicit creation of temporary arrays"],"relevant_pdf_pages":[232]},
    {"id":"q13","type":"conceptual","question":"How do resample() and asfreq() differ in Pandas time series?","ground_truth":"resample() is fundamentally a data aggregation, while asfreq() is fundamentally a data selection.","answerable":True,"anchors":["resample() is fundamentally a data aggregation, while asfreq() is fundamentally a data selection"],"relevant_pdf_pages":[216]},
    {"id":"q14","type":"factual","question":"What are the two interfaces Matplotlib provides?","ground_truth":"A convenient MATLAB-style state-based interface and a more powerful object-oriented interface.","answerable":True,"anchors":["a convenient MATLAB-style state-based interface, and a more powerful object-oriented interface"],"relevant_pdf_pages":[240]},
    {"id":"q15","type":"keyword","question":"How do I save a Matplotlib figure as a PNG file?","ground_truth":"Call fig.savefig('my_figure.png'); the file format is inferred from the file extension.","answerable":True,"anchors":["fig.savefig('my_figure.png')"],"relevant_pdf_pages":[239]},
    {"id":"q16","type":"factual","question":"What are the steps of the typical Scikit-Learn estimator API workflow?","ground_truth":"1) Choose a model class by importing the estimator. 2) Choose hyperparameters by instantiating it. 3) Arrange data into a features matrix and target vector. 4) Fit with fit(). 5) Apply to new data with predict() (supervised) or transform()/predict() (unsupervised).","answerable":True,"anchors":["Choose a class of model by importing the appropriate estimator class"],"relevant_pdf_pages":[365]},
    {"id":"q17","type":"conceptual","question":"What does it mean for a model to underfit versus overfit the data?","ground_truth":"Underfitting (high bias): the model lacks the flexibility to capture the data's features. Overfitting (high variance): the model is so flexible it fits the noise and generalizes poorly.","answerable":True,"anchors":["said to underfit the data; that is, it does not have enough model flexibility"],"relevant_pdf_pages":[382]},
    {"id":"q18","type":"paraphrase","question":"How should categorical features such as neighborhood names be encoded for a model?","ground_truth":"Use one-hot encoding, which creates extra 0/1 columns indicating presence or absence of each category; DictVectorizer does this for a list of dictionaries.","answerable":True,"anchors":["one proven technique is to use one-hot encoding"],"relevant_pdf_pages":[394]},
    {"id":"q19","type":"keyword","question":"What is TF-IDF and how is it computed in Scikit-Learn?","ground_truth":"Term frequency-inverse document frequency weights word counts by how often words appear across documents; computed with TfidfVectorizer.","answerable":True,"anchors":["which weights the word counts by a measure of how often they appear in the documents"],"relevant_pdf_pages":[396]},
    {"id":"q20","type":"factual","question":"What are the advantages of naive Bayes classifiers?","ground_truth":"They are extremely fast for training and prediction, give straightforward probabilistic predictions, are often easily interpretable, and have few or no tunable parameters.","answerable":True,"anchors":["They are extremely fast for both training and prediction"],"relevant_pdf_pages":[407]},
    {"id":"q21","type":"conceptual","question":"What is the kernel trick in support vector machines?","ground_truth":"A fit on kernel-transformed data can be done implicitly, without ever building the full high-dimensional kernel projection; this is built into the SVM (e.g. the RBF kernel).","answerable":True,"anchors":["a fit on kernel-transformed data can be done implicitly"],"relevant_pdf_pages":[431]},
    {"id":"q22","type":"paraphrase","question":"How can principal component analysis be used to clean up noisy data?","ground_truth":"Components with variance much larger than the noise are barely affected by it, so reconstructing the data from only the largest principal components keeps the signal and discards the noise.","answerable":True,"anchors":["any components with variance much larger than the effect of the noise should be relatively unaffected"],"relevant_pdf_pages":[458]},
    {"id":"q23","type":"factual","question":"How does the k-means algorithm work?","ground_truth":"Expectation-maximization: guess some cluster centers, then repeat until converged: E-step assigns points to the nearest cluster center, M-step sets each center to the mean of its assigned points.","answerable":True,"anchors":["E-Step: assign points to the nearest cluster center"],"relevant_pdf_pages":[483]},
    {"id":"q24","type":"conceptual","question":"Why might Gaussian mixture models be preferred over k-means?","ground_truth":"k-means has no intrinsic measure of probability or uncertainty of cluster assignments, and its cluster models must be circular (no way to handle oblong or elliptical clusters). GMM gives probabilistic assignments and fits elliptical clusters.","answerable":True,"anchors":["no intrinsic measure of probability or uncertainty of cluster assignments","these cluster models must be circular"],"relevant_pdf_pages":[495,497]},
    {"id":"q25","type":"keyword","question":"What does the bandwidth parameter control in kernel density estimation?","ground_truth":"The kernel bandwidth controls the size of the kernel at each point; it is a free parameter that can be chosen with cross-validation.","answerable":True,"anchors":["kernel bandwidth, which controls the size of the kernel at each point"],"relevant_pdf_pages":[514]},
    {"id":"q26","type":"multi_chunk","question":"What steps are involved in computing HOG features for images?","ground_truth":"1) Optionally prenormalize images. 2) Convolve with two filters sensitive to horizontal and vertical brightness gradients. 3) Subdivide into cells and compute a histogram of gradient orientations per cell. 4) Normalize the histograms in each cell by comparing to the block of neighboring cells. 5) Construct a one-dimensional feature vector from the information in each cell.","answerable":True,"anchors":["HOG involves the following steps","Construct a one-dimensional feature vector from the information in each cell"],"require_all_anchors":True,"relevant_pdf_pages":[524,525]},
    {"id":"q27","type":"conceptual","question":"How is bagging related to random forests?","ground_truth":"Bagging averages an ensemble of parallel overfit estimators to get a better classification; an ensemble of randomized decision trees is known as a random forest.","answerable":True,"anchors":["An ensemble of randomized decision trees is known as a random forest"],"relevant_pdf_pages":[444]},
    {"id":"q28","type":"paraphrase","question":"What three skill areas does the author say data science combines?","ground_truth":"Statistics (modeling and summarizing data), computer science (algorithms to store, process and visualize data), and domain expertise.","answerable":True,"anchors":["Data science comprises three distinct and overlapping areas"],"relevant_pdf_pages":[14]},
    {"id":"q29","type":"factual","question":"Does the book use Python 2 or Python 3 syntax?","ground_truth":"Python 3 syntax; most snippets will also work in Python 2, and incompatibilities are noted.","answerable":True,"anchors":["This book uses the syntax of Python 3"],"relevant_pdf_pages":[15]},
    {"id":"q30","type":"factual","question":"Which Python library is described as handling heterogeneous and labeled data?","ground_truth":"Pandas.","answerable":True,"anchors":["Pandas for manipulation of heterogeneous and labeled data"],"relevant_pdf_pages":[15]},
    {"id":"q31","type":"multi_chunk","question":"What are the weaknesses of k-means that motivate Gaussian mixture models, and how do GMMs generalize the E-M algorithm?","ground_truth":"k-means has no probabilistic measure of assignment uncertainty and assumes circular clusters; GMM generalizes E-M by replacing hard assignments with probabilistic (soft) assignments using Gaussian components.","answerable":True,"anchors":["no intrinsic measure of probability or uncertainty of cluster assignments","Generalizing E–M: Gaussian Mixture Models"],"require_all_anchors":True,"relevant_pdf_pages":[495,498]},
    {"id":"n01","type":"negative","question":"How does backpropagation train a deep neural network?","ground_truth":"NOT IN DOCUMENT. The book mentions deep neural networks only in passing (face detection section) and never explains backpropagation. The system should say it cannot find this.","answerable":False,"anchors":[],"relevant_pdf_pages":[]},
    {"id":"n02","type":"negative","question":"How does the self-attention mechanism in transformer models work?","ground_truth":"NOT IN DOCUMENT. 'Transformer' appears only as scikit-learn transformers (hard negative: keyword overlap).","answerable":False,"anchors":[],"relevant_pdf_pages":[]},
    {"id":"n03","type":"negative","question":"How do I install PyTorch and train a model on a GPU?","ground_truth":"NOT IN DOCUMENT. PyTorch is never mentioned.","answerable":False,"anchors":[],"relevant_pdf_pages":[]}
]


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def context_text(context):
    if isinstance(context, dict):
        return context.get("text", "")
    return context


def check(item, contexts):
    keyed = [normalized(context_text(context)) for context in contexts]
    ranks = [next((i + 1 for i, context in enumerate(keyed) if normalized(anchor) in context), None)
             for anchor in item["anchors"]]
    found = [rank for rank in ranks if rank is not None]
    if item.get("require_all_anchors"):
        rank = max(found) if len(found) == len(ranks) else None
    else:
        rank = min(found) if found else None
    return rank, len(found), len(ranks)


def retrieval_stats(rows, k):
    count = len(rows)
    hits = lambda cutoff: sum(bool(row["rank"] and row["rank"] <= cutoff) for row in rows)
    return {
        "n": count,
        "hit@1": round(hits(1) / count, 2) if count else 0,
        "hit@3": round(hits(3) / count, 2) if count else 0,
        f"hit@{k}": round(hits(k) / count, 2) if count else 0,
        "mrr": round(sum(1 / row["rank"] for row in rows if row["rank"]) / count, 2) if count else 0,
        "avg_ms": round(sum(row["ms"] for row in rows) / count) if count else 0,
    }


def import_module(module_name):
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, ROOT)
    return importlib.import_module(module_name)


def run_retrieval(args):
    module = import_module(args.module)
    module.USE_RERANK = not args.no_rerank

    try:
        stats = module.get_index().describe_index_stats()
        namespaces = getattr(stats, "namespaces", None) or stats["namespaces"]
        entry = namespaces.get(args.session_id)
        vector_count = (getattr(entry, "vector_count", None) or entry["vector_count"]) if entry else 0
        print(f"Pinecone vectors in namespace {args.session_id}: {vector_count}")
        if vector_count == 0:
            sys.exit("STOP: no Pinecone vectors for this session. Check ingestion and the session id.")
    except SystemExit:
        raise
    except Exception as error:
        print(f"(preflight vector-count check skipped: {error})")

    bm25 = module.get_bm25(f"{args.user_id}_{args.session_id}")
    parents = module.get_parents(args.user_id, args.session_id)
    if bm25 is None or not parents:
        sys.exit("No BM25 data or parent chunks found. Upload a PDF in this session first.")
    print(f"Loaded {len(parents)} parent chunks | k={args.k} alpha={args.alpha} rerank={not args.no_rerank}\n")

    rows = []
    for item in DATASET:
        print(f"[{item['id']}] ({item['type']}) {item['question']}")
        started = time.time()
        contexts = module.retrive(item["question"], bm25, parents, top_k=args.k,
                                  alpha=args.alpha, session_id=args.session_id)
        elapsed_ms = round((time.time() - started) * 1000)
        rank, anchors_found, anchors_total = check(item, contexts)
        status = f"HIT at rank {rank}" if rank else f"MISS ({anchors_found}/{anchors_total} evidence found)"
        print(f"   retrieved {len(contexts)} chunks in {elapsed_ms}ms -> {status}")
        if args.verbose or (item["answerable"] and not rank):
            for index, context in enumerate(contexts, 1):
                text = context_text(context)
                mark = "*" if any(normalized(anchor) in normalized(text) for anchor in item["anchors"]) else " "
                print(f"     {mark} #{index}: {text[:110].replace(chr(10), ' ')}")
        rows.append({
            "id": item["id"], "type": item["type"], "answerable": item["answerable"],
            "question": item["question"], "reference": item["ground_truth"],
            "retrieved_contexts": contexts, "anchor_rank": rank if item["answerable"] else None,
            "rank": rank, "anchors_found": anchors_found, "anchors_total": anchors_total,
            "ms": elapsed_ms, "n_contexts": len(contexts),
            "expected_pdf_pages": item["relevant_pdf_pages"],
        })
        time.sleep(args.sleep)
        if len(rows) == 3 and all(not row["retrieved_contexts"] for row in rows):
            sys.exit("STOP: the first three queries returned no contexts. Check vectors and namespace; nothing saved.")

    answerable = [row for row in rows if row["answerable"]]
    overall = retrieval_stats(answerable, args.k)
    by_type = defaultdict(list)
    for row in answerable:
        by_type[row["type"]].append(row)
    print("\nRetrieval summary:")
    print(json.dumps(overall))
    for question_type, group in by_type.items():
        summary = retrieval_stats(group, args.k)
        print(f"  {question_type}: n={summary['n']} hit@1={summary['hit@1']:.2f} "
              f"hit@{args.k}={summary[f'hit@{args.k}']:.2f} mrr={summary['mrr']:.2f}")
    misses = [row for row in answerable if not row["rank"]]
    print(f"Missed {len(misses)}/{len(answerable)} answerable questions.")

    result = {
        "config": {"k": args.k, "alpha": args.alpha, "rerank": not args.no_rerank},
        "overall": overall, "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
    print(f"Saved retrieval data to {args.out}")


def run_ragas(args):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
        from ragas.run_config import RunConfig
    except ImportError as error:
        sys.exit(f"RAGAS dependencies missing or incompatible: {error}\nSee the separate-venv instructions in this file.")

    with open(args.input, encoding="utf-8") as source:
        run = json.load(source)
    rows = [row for row in run["rows"] if row["answerable"]]
    empty = [row["id"] for row in rows if not row["retrieved_contexts"]]
    if len(empty) == len(rows):
        sys.exit("STOP: no retrieved contexts are available to score.")
    if empty:
        print(f"Skipping questions with no contexts: {empty}")
        rows = [row for row in rows if row["retrieved_contexts"]]
    if args.limit:
        rows = rows[:args.limit]

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not set.")
    from langchain_groq import ChatGroq
    try:
        llm = ChatGroq(model=args.judge_model, temperature=0, reasoning_effort="low")
    except Exception:
        llm = ChatGroq(model=args.judge_model, temperature=0)
    judge = LangchainLLMWrapper(llm)
    samples = [SingleTurnSample(user_input=row["question"],
                                retrieved_contexts=[context_text(context) for context in row["retrieved_contexts"]],
                                reference=row["reference"]) for row in rows]
    print(f"Scoring {len(samples)} questions with RAGAS (judge: {args.judge_model}) ...")
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=judge,
        run_config=RunConfig(max_workers=args.workers, timeout=args.timeout, max_retries=6),
        raise_exceptions=False,
        show_progress=True,
    )
    frame = result.to_pandas()
    precision_column = next(column for column in frame.columns if "precision" in column)
    recall_column = next(column for column in frame.columns if "recall" in column)
    per_question = []
    for index, row in enumerate(rows):
        per_question.append({
            "id": row["id"], "type": row["type"], "question": row["question"],
            "context_precision": None if frame[precision_column].isna()[index] else round(float(frame[precision_column][index]), 3),
            "context_recall": None if frame[recall_column].isna()[index] else round(float(frame[recall_column][index]), 3),
            "anchor_rank": row["anchor_rank"],
        })

    def mean(values):
        present = [value for value in values if value is not None]
        return round(sum(present) / len(present), 3) if present else None

    overall = {
        "n": len(per_question),
        "context_precision": mean([row["context_precision"] for row in per_question]),
        "context_recall": mean([row["context_recall"] for row in per_question]),
        "failed_scores": sum(row["context_precision"] is None or row["context_recall"] is None
                             for row in per_question),
    }
    by_type = defaultdict(list)
    for row in per_question:
        by_type[row["type"]].append(row)
    report = {
        "config": run.get("config", {}), "judge": args.judge_model, "overall": overall,
        "by_type": {kind: {
            "n": len(group),
            "context_precision": mean([row["context_precision"] for row in group]),
            "context_recall": mean([row["context_recall"] for row in group]),
        } for kind, group in by_type.items()},
        "per_question": per_question,
    }
    print("\nRAGAS retrieval scores:")
    print(json.dumps(overall))
    for question_type, group in by_type.items():
        print(f"  {question_type}: precision={mean([row['context_precision'] for row in group])} "
              f"recall={mean([row['context_recall'] for row in group])}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2, ensure_ascii=False)
        print(f"Saved RAGAS results to {args.out}")


def fetch_parent_ids(supabase, session_id, page_size=1000):
    ids, start = [], 0
    while True:
        result = (supabase.table("parent_store").select("id").eq("session_id", session_id)
                  .order("id").range(start, start + page_size - 1).execute())
        ids.extend(row["id"] for row in result.data)
        if len(result.data) < page_size:
            return ids
        start += page_size


def run_cleanup(args):
    module = import_module(args.module)
    result = module.get_index().query(vector=[0.0] * 1024, top_k=1000,
                                      include_metadata=True, namespace=args.session_id)
    document_ids = sorted({match["metadata"]["document_id"] for match in result["matches"]})
    if not document_ids:
        sys.exit("STOP: Pinecone has no vectors for this session. Nothing deleted.")
    print(f"Live document ids in Pinecone: {document_ids}")

    all_ids = fetch_parent_ids(module.supabase, args.session_id)
    live_ids = [parent_id for parent_id in all_ids
                if any(parent_id.startswith(document_id + "_p") for document_id in document_ids)]
    live_set = set(live_ids)
    orphan_ids = [parent_id for parent_id in all_ids if parent_id not in live_set]
    print(f"parent_store rows: {len(all_ids)} | live: {len(live_ids)} | orphans: {len(orphan_ids)}")
    if not live_ids:
        sys.exit("STOP: no live parent rows matched. Not deleting anything.")
    if not orphan_ids:
        print("Nothing to clean up.")
        return
    if not args.yes:
        print("DRY RUN: nothing deleted. Add --yes to delete the orphan rows.")
        return

    for start in range(0, len(orphan_ids), 200):
        module.supabase.table("parent_store").delete().in_("id", orphan_ids[start:start + 200]).execute()
    remaining = len(fetch_parent_ids(module.supabase, args.session_id))
    print(f"Deleted {len(orphan_ids)} orphan rows. Rows remaining: {remaining}")


def main():
    parser = argparse.ArgumentParser(description="Retrieval, RAGAS, and parent-store evaluation tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieval = subparsers.add_parser("retrieval", help="run retrieval checks on the embedded golden dataset")
    retrieval.add_argument("--module", default=DEFAULT_MODULE)
    retrieval.add_argument("--user-id", required=True)
    retrieval.add_argument("--session-id", required=True)
    retrieval.add_argument("--k", type=int, default=5)
    retrieval.add_argument("--alpha", type=float, default=0.5)
    retrieval.add_argument("--no-rerank", action="store_true")
    retrieval.add_argument("--sleep", type=float, default=0.2)
    retrieval.add_argument("--verbose", "-v", action="store_true")
    retrieval.add_argument("--out", default=DEFAULT_RUN)
    retrieval.set_defaults(handler=run_retrieval)

    ragas = subparsers.add_parser("ragas", help="score a saved retrieval run (requires the RAGAS venv)")
    ragas.add_argument("--input", default=DEFAULT_RUN)
    ragas.add_argument("--judge-model", default="openai/gpt-oss-120b")
    ragas.add_argument("--limit", type=int, default=0)
    ragas.add_argument("--workers", type=int, default=2)
    ragas.add_argument("--timeout", type=int, default=180)
    ragas.add_argument("--out", default=None, help="optional JSON report path")
    ragas.set_defaults(handler=run_ragas)

    cleanup = subparsers.add_parser("cleanup", help="inspect or remove orphaned parent-store rows")
    cleanup.add_argument("--module", default=DEFAULT_MODULE)
    cleanup.add_argument("--session-id", required=True)
    cleanup.add_argument("--yes", action="store_true", help="delete orphan rows; otherwise only report them")
    cleanup.set_defaults(handler=run_cleanup)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
