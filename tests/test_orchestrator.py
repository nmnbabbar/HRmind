import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.state import make_initial_state, PlannerOutput, AgentResult
from backend.orchestration.planner import planner_node
from backend.orchestration.router import router_node
from backend.orchestration.combiner import combiner_node

@pytest.mark.asyncio
async def test_router_sequential_execution():
    state = make_initial_state("test query", "session1")
    state["plan"] = {
        "agents": ["doc_parser", "sql"],
        "queries": {"doc_parser": "parse doc", "sql": "query sql"},
        "parallel": False,
        "reasoning": "Sequential test"
    }
    
    mock_doc_agent = AsyncMock()
    mock_doc_agent.run.return_value = AgentResult(
        agent_name="DocParserAgent",
        success=True,
        answer="Parsed doc",
        structured_data={"field": "value"},
        metadata={"entity_store": {"employee_id": "123"}}
    )
    
    mock_sql_agent = AsyncMock()
    mock_sql_agent.run.return_value = AgentResult(
        agent_name="SQLAgent",
        success=True,
        answer="Salary is 5000",
    )
    
    def get_agent_side_effect(name):
        if name == "doc_parser": return mock_doc_agent
        if name == "sql": return mock_sql_agent
        raise ValueError(name)
        
    with patch("backend.orchestration.router.AgentFactory.get_agent", side_effect=get_agent_side_effect):
        update = await router_node(state)
        
    assert "agent_results" in update
    assert len(update["agent_results"]) == 2
    
    # Verify sequential handoff works by checking if entity_store & parsed_document are updated in output
    assert "entity_store" in update
    assert update["entity_store"] == {"employee_id": "123"}
    assert "parsed_document" in update
    assert update["parsed_document"] == {"field": "value"}
    assert update["uploaded_file_path"] is None
    
@pytest.mark.asyncio
async def test_combiner_with_warnings():
    state = make_initial_state("test query", "session1")
    
    # Doc parser with low completeness score
    res1 = AgentResult(
        agent_name="DocParserAgent",
        success=True,
        answer="Parsed doc",
        metadata={"completeness_score": 0.2}
    )
    
    state["agent_results"] = [res1.to_dict()]
    
    with patch("backend.orchestration.combiner.ChatOpenAI") as mock_llm:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(content="Here is the information.")
        # Patch the pipe operator to return our mock chain
        mock_prompt = MagicMock()
        mock_prompt.__or__.return_value = mock_chain
        
        with patch("backend.orchestration.combiner.ChatPromptTemplate.from_messages", return_value=mock_prompt):
            update = combiner_node(state)
            
    assert "final_answer" in update
    assert "Here is the information." in update["final_answer"]
    assert "⚠️ Only partial information was extracted" in update["final_answer"]
