from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
import time
import markdown
# pyrefly: ignore[missing-import]
from bs4 import BeautifulSoup

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

def strip_markdown(text: str) -> str:
    # 1. Convert markdown to HTML
    html = markdown.markdown(text)
    # 2. Extract plain text from HTML
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()



def synthesizer_agent(state):
    print("synthesizer agent is working ....")
    messages = state["messages"]
    current_context = state.get("retrieved_context", [])
    pdf_paths = state.get("pdf_paths", [])
    t = state.get("routing_decision", {})
    tools = t.get('tools', [])
    print(tools)

    print(f"current_context length: {len(current_context)}")

    if not current_context:
        return {
            "synthesizer_result": "I couldn't find any relevant information to answer your question."
        }

    context_block = "\n\n---\n\n".join(current_context)

    doc_note = ""
    if pdf_paths or any(tool in tools for tool in ("hybrid_rag", "graph_rag")):
        doc_note = f"Note: The user HAS uploaded {len(pdf_paths)} document(s) ({', '.join(pdf_paths) if pdf_paths else 'via retrieval'}). The context below was retrieved from it. Never say no document is attached."

    system_prompt = f"""You are an expert synthesizer. Answer the user's question using ONLY the provided context.
        {doc_note}
        Rules:
        - Use ONLY the provided context
        - If contexts conflict, mention it
        - If context is insufficient, say so
        - Do not make up information
        - Be concise and direct
        - Do NOT claim no document is attached if context is provided below — the context below is your source of truth for this turn, not prior conversation turns

        <context>
        {context_block}
        </context>"""

    recent_messages = messages[-4:]
    full_messages = [SystemMessage(content=system_prompt)] + recent_messages

    t0 = time.time()
    result = llm.invoke(full_messages)
    clean_text = strip_markdown(result.content)
    elapsed = round(time.time() - t0, 2)
    print(f"⏱  [synthesizer] done in {elapsed}s")
    usage = result.response_metadata.get("token_usage", {})
    tokens = usage.get("total_tokens", 0) + state.get("total_tokens", 0)
    print(f"\nFinal Answer:\n{clean_text}")

    return {"synthesizer_result": clean_text, "total_tokens": tokens}