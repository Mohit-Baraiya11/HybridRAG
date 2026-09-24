import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import uuid
from supabase import create_client
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage

from graph.workflow import graph
from tools.hybrid_rag_tool import ingest_pdf
from tools.graph_rag_tool import ingest_pdf_for_graph
from fastapi.responses import FileResponse, HTMLResponse
from langchain_groq import ChatGroq
from fastapi.responses import StreamingResponse
import asyncio
import time

load_dotenv()

app = FastAPI()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

import re

@app.middleware("http")
async def normalize_path_middleware(request, call_next):
    if "//" in request.scope["path"]:
        request.scope["path"] = re.sub(r'/+', '/', request.scope["path"])
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","null"],  # tighten this after testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth & Database Client Helper ─────────────────────────────────────────────

async def get_user_db(authorization: str = Header(...)):
    try:
        token = authorization.replace("Bearer ", "").strip()
        if not token or token == "undefined":
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        user_resp = supabase.auth.get_user(token)
        user = user_resp.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        # Create user-authenticated client for PostgREST calls to pass RLS checks
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


# ── Background ingestion ──────────────────────────────────────────────────────

def run_ingestion(tmp_path: str, user_id: str, session_id: str, filename: str, token: str = None):
    try:
        ingest_pdf(tmp_path, user_id=user_id, session_id=session_id)
        ingest_pdf_for_graph(tmp_path, user_id=user_id, session_id=session_id)
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


# ── Session Endpoints ─────────────────────────────────────────────────────────

@app.post("/sessions")
async def create_session(body: SessionCreate, auth=Depends(get_user_db)):
    user, token, db = auth
    session_id = str(uuid.uuid4())
    db.table("sessions").insert({
        "id": session_id,
        "user_id": user.id,
        "title": body.title,
    }).execute()
    return {"session_id": session_id, "title": body.title}


@app.get("/sessions")
async def get_sessions(auth=Depends(get_user_db)):
    user, token, db = auth
    result = db.table("sessions") \
        .select("id, title, created_at, updated_at") \
        .eq("user_id", user.id) \
        .order("updated_at", desc=True) \
        .execute()
    return {"sessions": result.data}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, auth=Depends(get_user_db)):
    user, token, db = auth
    # Verify session belongs to user
    session = db.table("sessions") \
        .select("id") \
        .eq("id", session_id) \
        .eq("user_id", user.id) \
        .execute()

    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete session — messages cascade automatically
    db.table("sessions").delete().eq("id", session_id).execute()

    # Clean up related data
    db.table("messages").delete().eq("session_id", session_id).execute()
    db.table("bm25_store").delete().eq("id", f"{user.id}_{session_id}").execute()
    db.table("parent_store").delete().eq("session_id", session_id).execute()
    db.table("graph_store").delete().eq("id", f"{user.id}_{session_id}").execute()
    db.table("documents").delete().eq("session_id", session_id).execute()

    return {"status": "deleted", "session_id": session_id}


# ── Message Endpoints ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, auth=Depends(get_user_db)):
    user, token, db = auth
    # Verify session belongs to user
    session = db.table("sessions") \
        .select("id") \
        .eq("id", session_id) \
        .eq("user_id", user.id) \
        .execute()

    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found")

    result = db.table("messages") \
        .select("role, content, created_at") \
        .eq("session_id", session_id) \
        .order("created_at") \
        .execute()

    return {"messages": result.data}


# ── Upload Endpoint ───────────────────────────────────────────────────────────

@app.post("/upload/{session_id}")
async def upload_pdf(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auth=Depends(get_user_db)
):
    user, token, db = auth
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Verify session belongs to user
    session = db.table("sessions") \
        .select("id") \
        .eq("id", session_id) \
        .eq("user_id", user.id) \
        .execute()

    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Run ingestion in background
    background_tasks.add_task(
        run_ingestion,
        tmp_path=tmp_path,
        user_id=user.id,
        session_id=session_id,
        filename=file.filename,
        token=token
    )

    return {
        "status": "processing",
        "filename": file.filename,
        "message": "PDF uploaded. Ingestion running in background."
    }


# ── Chat Endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest, auth=Depends(get_user_db)):
    user, token, db = auth
    # Verify session belongs to user
    session = db.table("sessions") \
        .select("id") \
        .eq("id", request.session_id) \
        .eq("user_id", user.id) \
        .execute()

    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if first message
    msg_count = db.table("messages") \
        .select("id", count="exact") \
        .eq("session_id", request.session_id) \
        .execute()
    is_first_message = msg_count.count == 0

    # Fetch history
    history = db.table("messages") \
        .select("role, content") \
        .eq("session_id", request.session_id) \
        .order("created_at", desc=True) \
        .limit(7) \
        .execute()

    messages = []
    for msg in reversed(history.data):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Save user message
    db.table("messages").insert({
        "session_id": request.session_id,
        "user_id": user.id,
        "role": "user",
        "content": request.message
    }).execute()

    messages.append(HumanMessage(content=request.message))

    graph_start = time.time()

    # before graph.invoke
    docs = db.table("documents") \
        .select("filename") \
        .eq("session_id", request.session_id) \
        .execute()
    pdf_paths = [d["filename"] for d in docs.data]

    result = graph.invoke({
        "messages": messages,
        "history": history,
        "retrieved_context": [],
        "current_context": [],
        "user_id": user.id,
        "session_id": request.session_id,
        "verification_attempts": 0,
        "loop_decision": "",
        "routing_decision": {},
        "final_answer": "",
        "total_tokens": 0,
        "pdf_paths": pdf_paths
    })
    graph_latency = round(time.time() - graph_start, 2)
    print(f"Graph latency: {graph_latency}s")

    answer = result.get("synthesizer_result", "I couldn't generate an answer.")
    tokens = result.get("total_tokens", 0)

    # Save assistant message
    db.table("messages").insert({
        "session_id": request.session_id,
        "user_id": user.id,
        "role": "assistant",
        "content": answer,
    }).execute()

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    # Generate title on first message only
    if is_first_message:
        title_result = llm.invoke(f"Summarize this message into a short 3-5 word chat title:\n\n{request.message}")
        title = title_result.content.strip()
        db.table("sessions").update({
            "title": title,
            "updated_at": "now()",
            "total_tokens": tokens
        }).eq("id", request.session_id).execute()
    else:
        current = db.table("sessions") \
            .select("total_tokens") \
            .eq("id", request.session_id) \
            .execute()
        current_tokens = current.data[0].get("total_tokens", 0) or 0
        db.table("sessions").update({
            "updated_at": "now()",
            "total_tokens": current_tokens + tokens,
            "last_latency": graph_latency
        }).eq("id", request.session_id).execute()

    return {"answer": answer, "session_id": request.session_id, "tokens_used": tokens}


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def check():
    return {"message": "API is running......"}