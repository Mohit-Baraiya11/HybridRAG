import os
import re
import time
import uuid
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()  # must run before anything reads env vars
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from supabase import create_client
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

from tools.hybrid_rag_tool import (
    ingest_pdf,
    warm_up,
    retrieve_context,
    get_index,
    delete_session_vectors,
    clear_session_caches,
)

# Optional: verify the JWT locally (no network call per request).
# Needs `pip install "pyjwt[crypto]"`. If anything fails we fall back to
# supabase.auth.get_user(), so auth keeps working either way.
try:
    import jwt
    from jwt import PyJWKClient
    _jwks_client = PyJWKClient(
        f"{os.getenv('SUPABASE_URL')}/auth/v1/.well-known/jwks.json", cache_keys=True
    )
except Exception:
    jwt = None
    _jwks_client = None

app = FastAPI()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
admin_db = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
title_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, reasoning_effort="low")

SYSTEM_PROMPT = """You are a strict document-only assistant. Your ONLY job is to extract answers directly from the provided Passages.

CRITICAL RULES:
1. STRICT ISOLATION: Treat your own memory as completely wiped. You know NOTHING about the world, definitions, concepts, or facts outside of the Passages below.
2. EXACT EVIDENCE REQUIRED: Just because a keyword appears in a Passage (for example, a skill listed on a resume) does NOT mean you can define or explain that concept. If the Passage does not explicitly explain or answer the user's question, you MUST refuse to answer.
3. MANDATORY REFUSAL: If the Passages are empty, say "No relevant passages were found", or do not contain the exact answer, respond ONLY with:
"I couldn't find that in your documents."
(You may add one short sentence suggesting a topic that IS in the passages. Do NOT attempt to answer the user's question.)
4. CITATIONS: Add the passage number in square brackets after every statement, like. Do not use square brackets for anything else.
5. CONVERSATION: For basic greetings ("hi", "thanks") or questions about previous messages in this chat, reply briefly and remind the user you only answer questions about their uploaded documents.

Passages:
{context}

REMINDER: If the answer is not explicitly written in the Passages above, output "I couldn't find that in your documents." and nothing else."""


NO_CONTEXT = "(No relevant passages were found. The documents may not be uploaded or indexed yet.)"


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return NO_CONTEXT
    return "\n\n".join(f"[{i}]\n{c['text']}" for i, c in enumerate(chunks, 1))


def process_answer(answer: str, chunks: list[dict]):
    """Remove citation markers that don't match a passage, and return every
    retrieved passage (with its page) flagged by whether the answer cited it."""
    cited = set()

    def keep(m):
        n = int(m.group(1))
        if 1 <= n <= len(chunks):
            cited.add(n)
            return m.group(0)
        return ""

    cleaned = re.sub(r"\[(\d{1,2})\]", keep, answer)

    # A refusal with nothing cited shouldn't be dressed up with unrelated passages
    if not cited and "couldn't find" in cleaned.lower():
        return cleaned, []

    sources = [
        {"n": i, "source": c["source"], "page": c["page"], "text": c["text"], "cited": i in cited}
        for i, c in enumerate(chunks, 1)
    ]
    return cleaned, sources


def strip_citations(text: str) -> str:
    """Old [n] markers point at passages from earlier turns; don't feed them back."""
    return re.sub(r"\s?\[\d{1,2}\]", "", text)


@app.on_event("startup")
def _startup():
    warm_up()


@app.middleware("http")
async def normalize_path_middleware(request, call_next):
    if "//" in request.scope["path"]:
        request.scope["path"] = re.sub(r"/+", "/", request.scope["path"])
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],  # tighten this after testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth & Database Client Helper ─────────────────────────────────────────────

def _verify_locally(token: str):
    """Returns a user-like object, or None if local verification isn't possible."""
    if jwt is None or _jwks_client is None:
        return None
    try:
        key = _jwks_client.get_signing_key_from_jwt(token).key
        payload = jwt.decode(
            token, key, algorithms=["ES256", "RS256"], audience="authenticated"
        )
        return SimpleNamespace(id=payload["sub"])
    except Exception:
        return None


def get_user_db(authorization: str = Header(...)):
    try:
        token = authorization.replace("Bearer ", "").strip()
        if not token or token == "undefined":
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        user = _verify_locally(token)
        if user is None:  # fallback: ask Supabase
            user = supabase.auth.get_user(token).user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        # User-authenticated client so PostgREST calls pass RLS checks
        db = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        db.postgrest.auth(token)
        return user, token, db
    except HTTPException:
        raise
    except Exception as e:
        print(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str


class SessionCreate(BaseModel):
    title: str = "New Chat"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _owns_session(db, user_id: str, session_id: str) -> bool:
    res = db.table("sessions").select("id") \
        .eq("id", session_id).eq("user_id", user_id).execute()
    return bool(res.data)


# ── Background jobs ───────────────────────────────────────────────────────────

def run_ingestion(tmp_path: str, user_id: str, session_id: str, filename: str, token: str = None):
    try:
        ingest_pdf(tmp_path, user_id=user_id, session_id=session_id, filename=filename)
        print(f"Background ingestion complete: {filename}")

        db = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        if token:
            db.postgrest.auth(token)

        db.table("documents").insert({
            "session_id": session_id,
            "user_id": user_id,
            "filename": filename,
        }).execute()

    except Exception as e:
        print(f"Background ingestion failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"Temp file deleted: {tmp_path}")


def post_chat_tasks(user_id: str, session_id: str, first_message: str,
                    is_first: bool, tokens: int, latency: float):
    """Everything that doesn't need to block the answer: title, session stats, latency log."""
    try:
        update = {"updated_at": "now()", "last_latency": latency}

        if is_first:
            title = title_llm.invoke(
                f"Summarize this message into a short 3-5 word chat title:\n\n{first_message}"
            ).content.strip()
            update["title"] = title

        cur = admin_db.table("sessions").select("total_tokens").eq("id", session_id).execute()
        current_tokens = (cur.data[0].get("total_tokens") or 0) if cur.data else 0
        update["total_tokens"] = current_tokens + tokens

        admin_db.table("sessions").update(update).eq("id", session_id).execute()

        admin_db.table("latency_logs").insert({
            "user_id": user_id,
            "session_id": session_id,
            "latency_seconds": latency,
            "tokens_used": tokens,
        }).execute()
    except Exception as e:
        print(f"post_chat_tasks failed: {e}")


# ── Session Endpoints ─────────────────────────────────────────────────────────

@app.post("/sessions")
def create_session(body: SessionCreate, auth=Depends(get_user_db)):
    user, token, db = auth
    session_id = str(uuid.uuid4())
    db.table("sessions").insert({
        "id": session_id,
        "user_id": user.id,
        "title": body.title,
    }).execute()
    return {"session_id": session_id, "title": body.title}


@app.get("/sessions")
def get_sessions(auth=Depends(get_user_db)):
    user, token, db = auth
    result = db.table("sessions") \
        .select("id, title, created_at, updated_at") \
        .eq("user_id", user.id) \
        .order("updated_at", desc=True) \
        .execute()
    return {"sessions": result.data}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, auth=Depends(get_user_db)):
    user, token, db = auth
    if not _owns_session(db, user.id, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Vectors + side tables first. Service role is used for the tables that are
    # written with the service role, because RLS can make user-client deletes
    # silently remove 0 rows.
    delete_session_vectors(session_id)
    admin_db.table("bm25_store").delete().eq("id", f"{user.id}_{session_id}").execute()
    admin_db.table("parent_store").delete().eq("session_id", session_id).execute()
    admin_db.table("documents").delete().eq("session_id", session_id).execute()
    admin_db.table("messages").delete().eq("session_id", session_id).execute()
    admin_db.table("sessions").delete().eq("id", session_id).eq("user_id", user.id).execute()

    clear_session_caches(user.id, session_id)
    return {"status": "deleted", "session_id": session_id}


# ── Message Endpoints ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, auth=Depends(get_user_db)):
    user, token, db = auth
    if not _owns_session(db, user.id, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    result = db.table("messages") \
        .select("role, content, sources, created_at") \
        .eq("session_id", session_id) \
        .order("created_at") \
        .execute()

    return {"messages": result.data}


# ── Upload Endpoint ───────────────────────────────────────────────────────────

@app.post("/upload/{session_id}")
def upload_pdf(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auth=Depends(get_user_db),
):
    user, token, db = auth
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    if not _owns_session(db, user.id, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    background_tasks.add_task(
        run_ingestion,
        tmp_path=tmp_path,
        user_id=user.id,
        session_id=session_id,
        filename=file.filename,
        token=token,
    )

    return {
        "status": "processing",
        "filename": file.filename,
        "message": "PDF uploaded. Ingestion running in background.",
    }


# ── Chat Endpoint ─────────────────────────────────────────────────────────────
# Plain `def` on purpose: FastAPI runs it in a threadpool, so the blocking
# Supabase / Pinecone / Groq calls don't freeze the event loop for other users.

@app.post("/chat")
def chat(request: ChatRequest, background_tasks: BackgroundTasks, auth=Depends(get_user_db)):
    user, token, db = auth
    t0 = time.time()

    if not _owns_session(db, user.id, request.session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Recent history (newest first -> chronological). Empty means first message.
    history = db.table("messages") \
        .select("role, content") \
        .eq("session_id", request.session_id) \
        .order("created_at", desc=True) \
        .limit(6) \
        .execute().data
    is_first = len(history) == 0

    past = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=strip_citations(m["content"]))
        for m in reversed(history)
    ]

    db.table("messages").insert({
        "session_id": request.session_id,
        "user_id": user.id,
        "role": "user",
        "content": request.message,
    }).execute()

    # Retrieve -> single LLM call
    chunks = retrieve_context(request.message, user.id, request.session_id)

    with_context = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=build_context(chunks))),
        *past,
        HumanMessage(content=request.message),
    ]
    resp = llm.invoke(with_context)
    answer, sources = process_answer(resp.content or "I couldn't generate an answer.", chunks)
    tokens = (getattr(resp, "usage_metadata", None) or {}).get("total_tokens", 0)

    db.table("messages").insert({
        "session_id": request.session_id,
        "user_id": user.id,
        "role": "assistant",
        "content": answer,
        "sources": sources,
    }).execute()

    latency = round(time.time() - t0, 2)
    print(f"Chat latency: {latency}s")

    background_tasks.add_task(
        post_chat_tasks, user.id, request.session_id, request.message, is_first, tokens, latency
    )

    return {"answer": answer, "sources": sources, "session_id": request.session_id, "tokens_used": tokens}


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api-status")
def check():
    return {"message": "API is running......"}


@app.get("/")
def serve_ui():
    """Serve the frontend UI from the same origin as the API."""
    return FileResponse("index.html")


# ── Latency metrics ───────────────────────────────────────────────────────────

def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return round(s[f] + (s[c] - s[f]) * (k - f), 2)


def _summary(rows):
    v = [float(r["latency_seconds"]) for r in rows if r.get("latency_seconds") is not None]
    return {
        "count": len(v),
        "p50": _pct(v, 50),
        "p95": _pct(v, 95),
        "mean": round(sum(v) / len(v), 2) if v else None,
    }


@app.get("/metrics/latency")
def latency_metrics(days: int = 14, auth=Depends(get_user_db)):
    user, token, db = auth
    days = max(1, min(days, 90))
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = (admin_db.table("latency_logs")
            .select("user_id,latency_seconds,created_at")
            .gte("created_at", since).order("created_at", desc=True)
            .limit(5000).execute()).data or []
    mine = [r for r in rows if r["user_id"] == user.id]
    by_day = {}
    for r in mine:
        by_day.setdefault(r["created_at"][:10], []).append(r)
    return {
        "days": days,
        "me": _summary(mine),
        "all": {**_summary(rows), "users": len({r["user_id"] for r in rows})},
        "daily": [{"date": d, **_summary(v)} for d, v in sorted(by_day.items())],
        "recent": [{"at": r["created_at"], "seconds": r["latency_seconds"]} for r in mine[:10]],
    }