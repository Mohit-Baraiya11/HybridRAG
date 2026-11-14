from langgraph.graph import StateGraph,START,END
from schemas.state import AgentState

from tools.hybrid_rag_tool import hybrid_rag_node

from graph.routing import route_after_verifier, route_tools

from agents.synthesizer_agent import synthesizer_agent
from agents.verifier_agent import verifier_agent

builder = StateGraph(AgentState)
builder.add_node("hybrid_rag", hybrid_rag_node)
builder.add_node("synthesizer", synthesizer_agent)
builder.add_node("verifier", verifier_agent)

builder.add_edge(START,"hybrid_rag")

builder.add_edge("hybrid_rag", "synthesizer")
builder.add_edge("synthesizer","verifier")

builder.add_conditional_edges(
    "verifier",
    route_after_verifier,
    {
       "approve":END,
       "re_evaluate": "hybrid_rag",
    }

)

graph = builder.compile()
