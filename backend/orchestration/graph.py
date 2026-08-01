from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.state import GraphState
from backend.orchestration.planner import planner_node
from backend.orchestration.router import router_node
from backend.orchestration.combiner import combiner_node

def route_after_planner(state: GraphState) -> str:
    plan_dict = state.get("plan")
    if plan_dict and not plan_dict.get("agents"):
        return END
    return "router"

def route_after_router(state: GraphState) -> str:
    plan_dict = state.get("plan")
    if plan_dict:
        agents = plan_dict.get("agents", [])
        if len(agents) <= 1:
            return END
    return "combiner"

def build_graph():
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("planner", planner_node)
    builder.add_node("router", router_node)
    builder.add_node("combiner", combiner_node)
    
    # Add edges
    builder.set_entry_point("planner")
    builder.add_conditional_edges("planner", route_after_planner, {"router": "router", END: END})
    builder.add_conditional_edges("router", route_after_router, {"combiner": "combiner", END: END})
    builder.add_edge("combiner", END)
    
    # Compile with memory persistence
    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)
    
    return app

# Singleton export
graph_app = build_graph()
