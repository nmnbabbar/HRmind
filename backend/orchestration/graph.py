from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.state import GraphState
from backend.orchestration.planner import planner_node
from backend.orchestration.router import router_node
from backend.orchestration.combiner import combiner_node

def build_graph():
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("planner", planner_node)
    builder.add_node("router", router_node)
    builder.add_node("combiner", combiner_node)
    
    # Add edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "router")
    builder.add_edge("router", "combiner")
    builder.add_edge("combiner", END)
    
    # Compile with memory persistence
    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)
    
    return app

# Singleton export
graph_app = build_graph()
