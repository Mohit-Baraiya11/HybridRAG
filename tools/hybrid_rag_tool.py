from pathlib import Path
from contextlib import contextmanager
import uuid
import json as json
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from supabase import create_client
from pinecone_text.sparse import BM25Encoder

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
bm25 = BM25Encoder()   # only used during ingestion (train + encode_documents)

# in-memory parent store, only used while ingesting a PDF
parent_store = {}

# ── Runtime config + caches (query path) ──────────────────────────────────────
USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"

_index = None
_bm25_cache = {}     # bm25_id -> BM25Encoder
_parent_cache = {}   # (user_id, session_id) -> {parent_id: text}


@contextmanager
def timer(label):
    t = time.time()
    yield
    print(f"   ⏱ {label}: {round(time.time() - t, 2)}s")


def get_index():
    """Create/fetch the Pinecone index once per process."""
    global _index
    if _index is None:
        _index = create_index_if_not_exists()
    return _index


def get_bm25(bm25_id: str):
    """Cached BM25 encoder for a session. Returns None if not found."""
    if bm25_id in _bm25_cache:
        return _bm25_cache[bm25_id]
    result = supabase.table("bm25_store").select("params").eq("id", bm25_id).execute()
    if not result.data:
        print("No BM25 params found.")
        return None
    enc = BM25Encoder()
    enc.set_params(**json.loads(result.data[0]["params"]))
    _bm25_cache[bm25_id] = enc
    return enc


def get_parents(user_id: str, session_id: str) -> dict:
    """Cached parent chunks for a session."""
    key = (user_id, session_id)
    if key in _parent_cache:
        return _parent_cache[key]
    result = supabase.table("parent_store") \
        .select("id, content") \
        .eq("user_id", user_id) \
        .eq("session_id", session_id) \
        .execute()
    parents = {row["id"]: row["content"] for row in result.data}
    if parents:
        _parent_cache[key] = parents
    return parents


def warm_up():
    """Call once at server startup so the first real request isn't slow."""
    try:
        index = get_index()
        embedding_text(["warm up"], input_type="query")
        index.query(vector=[0.0] * 1024, top_k=1)
        print("warm-up done")
    except Exception as e:
        print(f"warm-up failed: {e}")


# ── Loading + chunking ────────────────────────────────────────────────────────

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()




def chunk_document(docs, document_id):
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    parent_chunks = parent_splitter.split_documents(docs)
    all_child_chunks = []

    for p_idx, parent_chunk in enumerate(parent_chunks):
        parent_id = f"{document_id}_p{p_idx}"
        parent_store[parent_id] = parent_chunk.page_content  # fill in-memory store

        children = child_splitter.split_documents([parent_chunk])
        for c_idx, child in enumerate(children):
            child.metadata["parent_id"] = parent_id
            child.metadata["chunk_id"] = f"{parent_id}_c{c_idx}"
            child.metadata["document_id"] = document_id
            all_child_chunks.append(child)

    return all_child_chunks


# ── Parent Store Supabase ─────────────────────────────────────────────────────

def save_parent_store_to_supabase(user_id: str, session_id: str, source: str):
    """Save current in-memory parent_store to Supabase."""
    rows = [
        {
            "id": pid,
            "content": text,
            "user_id": user_id,
            "session_id": session_id,
            "source": source
        }
        for pid, text in parent_store.items()
    ]
    if rows:
        supabase.table("parent_store").upsert(rows).execute()
        print(f"Parent store saved: {len(rows)} parents")


def load_parent_store_from_supabase(user_id: str, session_id: str):
    """Legacy loader (kept in case other modules import it). Query path uses get_parents()."""
    global parent_store
    result = supabase.table("parent_store") \
        .select("id, content") \
        .eq("user_id", user_id) \
        .eq("session_id", session_id) \
        .execute()

    if result.data:
        parent_store = {row["id"]: row["content"] for row in result.data}
        print(f"Parent store loaded: {len(parent_store)} parents")
    else:
        print("No parent store found in Supabase.")


# ── BM25 ──────────────────────────────────────────────────────────────────────

def train_and_save_bm25(texts: list[str], user_id: str = "default"):
    global bm25
    bm25.fit(texts)
    params = bm25.get_params()
    supabase.table("bm25_store").upsert({
        "id": user_id,
        "params": json.dumps(params)
    }).execute()
    print(f"BM25 saved for user: {user_id}")


def load_bm25_from_supabase(user_id="default"):
    """Legacy loader (kept in case other modules import it). Query path uses get_bm25()."""
    global bm25
    result = supabase.table("bm25_store") \
        .select("params") \
        .eq("id", user_id) \
        .execute()

    if result.data:
        params = json.loads(result.data[0]["params"])
        bm25 = BM25Encoder()
        bm25.set_params(**params)
        print(f"BM25 loaded for user: {user_id}")
        return True

    print("No BM25 params found.")
    return False


def get_session_texts_from_pinecone(session_id: str) -> list:
    index = get_index()
    all_texts = []

    results = index.query(
        vector=[0.0] * 1024,
        top_k=1000,
        include_metadata=True,
        namespace=session_id,
    )

    for match in results["matches"]:
        text = match["metadata"].get("text", "")
        if text:
            all_texts.append(text)

    return all_texts


# ── Embeddings ────────────────────────────────────────────────────────────────

def embedding_text(texts, input_type="passage", batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        # drop lone surrogate characters that some PDF math symbols produce
        batch = [t.encode("utf-8", "ignore").decode("utf-8") for t in texts[i:i + batch_size]]
        for attempt in range(6):
            try:
                res = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=batch,
                    parameters={"input_type": input_type, "truncate": "END"},
                )
                out.extend(list(res))
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"embed batch {i // batch_size} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"embedding failed for batch starting at {i}")
        if len(texts) > batch_size:
            print(f"embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return out


# ── Pinecone ──────────────────────────────────────────────────────────────────

def create_index_if_not_exists():
    index_name = os.getenv("PINECONE_INDEX_NAME")
    existing = [i.name for i in pc.indexes.list()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)


def pinecone_store(chunks, embeddings, session_id: str = "default"):
    index = get_index()
    texts = [chunk.page_content for chunk in chunks]
    sparse_embeddings = bm25.encode_documents(texts)
    vectors = []
    for chunk, dense_emb, sparse_emb in zip(chunks, embeddings, sparse_embeddings):
        vectors.append({
            "id": chunk.metadata["chunk_id"],
            "values": dense_emb["values"],
            "sparse_values": sparse_emb,
            "metadata": {
                "session_id": session_id,
                "document_id": chunk.metadata["document_id"],
                "chunk_id": chunk.metadata["chunk_id"],
                "parent_id": chunk.metadata["parent_id"],
                "source": chunk.metadata["source"],
                "page": chunk.metadata.get("page", 0),
                "text": chunk.page_content
            }
        })
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100], namespace=session_id)


def is_document_already_indexed(filename: str, session_id: str = "default") -> bool:
    index = get_index()
    results = index.query(
        vector=[0.0] * 1024,
        top_k=1,
        include_metadata=True,
        filter={"source": {"$eq": filename}},
        namespace=session_id,
    )
    return len(results["matches"]) > 0


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrive(query, bm25_enc, parents, top_k=5, alpha=0.5, source_filter=None, session_id: str = "default"):
    with timer("embed query"):
        dense_vector = embedding_text([query], input_type="query")[0]["values"]
    sparse_vector = bm25_enc.encode_queries(query)

    scaled_dense = [v * alpha for v in dense_vector]
    scaled_sparse = {
        "indices": sparse_vector["indices"],
        "values": [v * (1 - alpha) for v in sparse_vector["values"]]
    }

    query_kwargs = {
        "vector": scaled_dense,
        "sparse_vector": scaled_sparse,
        "top_k": 30 if USE_RERANK else top_k,
        "include_metadata": True,
        "namespace": session_id,          # isolates this session's vectors
    }
    if source_filter:
        query_kwargs["filter"] = {"source": {"$eq": source_filter}}

    index = get_index()                   # outside the timer on purpose
    with timer("pinecone query"):
        results = index.query(**query_kwargs)
    matches = results["matches"]

    if matches and USE_RERANK:
        with timer("rerank"):
            reranked = pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=[m["metadata"]["text"] for m in matches],
                top_n=min(len(matches), top_k * 3),
                return_documents=True
            )
        matches = [matches[r["index"]] for r in reranked.data]

    contexts = []
    seen_parents = set()
    for match in matches:
        parent_id = match["metadata"].get("parent_id")
        if parent_id and parent_id not in seen_parents:
            parent_text = parents.get(parent_id, match["metadata"].get("text", ""))
            contexts.append(parent_text)
            seen_parents.add(parent_id)
            if len(contexts) >= top_k:
                break

    return contexts


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str, user_id: str, session_id: str):

    filename = Path(pdf_path).name
    bm25_id = f"{user_id}_{session_id}"

    # Skip if already indexed
    if is_document_already_indexed(filename, session_id=session_id):
        print(f"'{filename}' already indexed, skipping.")
        return

    # Fresh parent_store for this ingestion, so parents from another
    # session/user never get upserted under this one
    parent_store.clear()

    docs = load_pdf(pdf_path)
    for d in docs:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8")
    document_id = f"doc_{uuid.uuid4().hex[:12]}"

    for doc in docs:
        doc.metadata["document_id"] = document_id
        doc.metadata["source"] = filename

    child_chunks = chunk_document(docs, document_id)
    existing_texts = get_session_texts_from_pinecone(session_id)
    new_texts = [chunk.page_content for chunk in child_chunks]
    all_texts = existing_texts + new_texts

    save_parent_store_to_supabase(user_id, session_id, filename)
    train_and_save_bm25(all_texts, user_id=bm25_id)

    texts = [chunk.page_content for chunk in child_chunks]
    dense_embedding = embedding_text(texts)
    pinecone_store(child_chunks, dense_embedding, session_id=session_id)

    # Invalidate query-side caches so the next question sees the new document
    _bm25_cache.pop(bm25_id, None)
    _parent_cache.pop((user_id, session_id), None)

    print(f"Ingestion complete for {filename}")


# ── LangGraph node ────────────────────────────────────────────────────────────

def hybrid_rag_node(state):
    user_query = state["messages"][-1].content
    user_id = state.get("user_id", "default")
    session_id = state.get("session_id", "default")

    t0 = time.time()
    with timer("bm25 + parents"):
        bm25_enc = get_bm25(f"{user_id}_{session_id}")
        parents = get_parents(user_id, session_id)

    if bm25_enc is None:
        context = []
    else:
        context = retrive(
            user_query,
            bm25_enc,
            parents,
            top_k=5,
            alpha=0.5,
            source_filter=None,
            session_id=session_id,
        )

    print(f"⏱  [hybrid_rag] done in {round(time.time() - t0, 2)}s  —  {len(context)} chunks retrieved")

    return {
        "retrieved_context": context,
        "current_context": context,
    }