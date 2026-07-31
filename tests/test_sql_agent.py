"""
tests/test_sql_agent.py
=======================
Tests for the SQLAgent and SQLValidator.
"""

import pytest
import aiosqlite
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch, MagicMock

from backend.agents.sql_agent.sql_validator import validate_and_format_sql, SQLValidationError
from backend.agents.sql_agent.sql_agent import SQLAgent
from backend.state import GraphState, PlannerOutput

# ── Validator Tests ────────────────────────────────────────────────────────────

def test_sql_validator_valid_select():
    sql = "SELECT first_name, last_name FROM employees WHERE department_id = 1"
    safe = validate_and_format_sql(sql, max_rows=100)
    assert "SELECT" in safe
    assert "LIMIT 100" in safe

def test_sql_validator_existing_limit():
    sql = "SELECT * FROM employees LIMIT 5"
    safe = validate_and_format_sql(sql, max_rows=100)
    assert "LIMIT 5" in safe

def test_sql_validator_exceeding_limit():
    sql = "SELECT * FROM employees LIMIT 500"
    safe = validate_and_format_sql(sql, max_rows=100)
    assert "LIMIT 100" in safe

def test_sql_validator_blocks_mutations():
    bad_queries = [
        "DROP TABLE employees",
        "DELETE FROM employees WHERE id = 1",
        "UPDATE salary_history SET salary_amount = 999999",
        "INSERT INTO departments (name) VALUES ('Hacked')",
        "ALTER TABLE employees ADD COLUMN ssn TEXT"
    ]
    for q in bad_queries:
        with pytest.raises(SQLValidationError, match="Only SELECT statements are allowed"):
            validate_and_format_sql(q)

def test_sql_validator_blocks_multi_statement_injection():
    # Attempting to sneak a DROP TABLE after a valid SELECT
    sneaky_query = "SELECT * FROM employees; DROP TABLE employees;"
    with pytest.raises(SQLValidationError, match="Multiple SQL statements are not allowed"):
        validate_and_format_sql(sneaky_query)

def test_sql_validator_blocks_subquery_mutation():
    # Attempting to sneak a mutation in a weird AST shape
    # sqlglot usually rejects non-selects outright in its AST if read as strict DML
    with pytest.raises(SQLValidationError):
        validate_and_format_sql("SELECT * FROM (DROP TABLE employees)")


# ── SQLAgent Tests ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_sql():
    mock = MagicMock()
    mock.ainvoke = AsyncMock()
    mock.ainvoke.return_value.content = "SELECT first_name, last_name FROM employees LIMIT 10"
    return mock

@pytest.fixture
def sample_sql_state():
    return GraphState(
        query="Who works in Engineering?",
        session_id="test-session",
        plan={
            "agents": ["sql"],
            "queries": {"sql": "Who works in Engineering?"},
            "parallel": False
        },
        agent_results=[],
        error=None
    )

@pytest.mark.asyncio
async def test_sql_agent_missing_plan_query(mock_llm_sql):
    agent = SQLAgent(llm=mock_llm_sql)
    # State with no plan
    result = await agent.run(GraphState(query="test", session_id="1"))
    assert not result.success
    assert "No SQL query found" in result.error

@pytest.mark.asyncio
async def test_sql_agent_llm_hallucinates_markdown(mock_llm_sql, sample_sql_state):
    # LLM hallucinates markdown tags
    mock_llm_sql.ainvoke.return_value.content = "```sql\nSELECT * FROM departments\n```"
    
    agent = SQLAgent(llm=mock_llm_sql)
    
    # Use a real in-memory DB instead of mocking aiosqlite context managers
    
    async with aiosqlite.connect(":memory:") as db:
        await db.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
        await db.execute("INSERT INTO departments (name) VALUES ('Engineering')")
        await db.commit()
        
        with patch("backend.agents.sql_agent.sql_agent.aiosqlite.connect") as mock_connect:
            # We mock the connect function to yield our in-memory DB
            @asynccontextmanager
            async def get_db(*args, **kwargs):
                yield db
            
            mock_connect.side_effect = get_db
            result = await agent.run(sample_sql_state)
            
    assert result.success
    assert "Engineering" in result.answer
    assert result.structured_data["sql_query"].startswith("SELECT")
    assert "```sql" not in result.structured_data["sql_query"]

@pytest.mark.asyncio
async def test_sql_agent_validation_failure(mock_llm_sql, sample_sql_state):
    # LLM returns a DROP statement
    mock_llm_sql.ainvoke.return_value.content = "DROP TABLE employees"
    
    agent = SQLAgent(llm=mock_llm_sql)
    result = await agent.run(sample_sql_state)
    
    assert not result.success
    assert "Unsafe or invalid SQL generated" in result.error
    assert "Only SELECT" in result.error

@pytest.mark.asyncio
async def test_sql_agent_success_formatting(mock_llm_sql, sample_sql_state):
    agent = SQLAgent(llm=mock_llm_sql)
    
    async with aiosqlite.connect(":memory:") as db:
        await db.execute("CREATE TABLE employees (first_name TEXT, last_name TEXT)")
        await db.execute("INSERT INTO employees VALUES ('Alice', 'Smith'), ('Bob', 'Jones')")
        await db.commit()
        
        with patch("backend.agents.sql_agent.sql_agent.aiosqlite.connect") as mock_connect:
            @asynccontextmanager
            async def get_db(*args, **kwargs):
                yield db
            
            mock_connect.side_effect = get_db
            result = await agent.run(sample_sql_state)
            
    assert result.success
    assert "Alice" in result.answer
    assert "| first_name | last_name |" in result.answer
    assert len(result.structured_data["rows"]) == 2
