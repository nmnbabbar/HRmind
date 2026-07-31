from backend.orchestration.graph import graph_app

def get_graph():
    """Dependency to provide the LangGraph app"""
    return graph_app
