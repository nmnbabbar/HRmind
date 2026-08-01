"""
backend/agents/sql_agent/sql_agent.py
=====================================
SQL Agent Implementation.

Translates Natural Language to SQL, strictly validates it using sqlglot,
executes it against the HR SQLite database, and returns the results.
"""

import sqlite3
import time
from typing import Any, cast

import aiosqlite
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from backend.base.agent import BaseAgent
from backend.base.guardrail import GuardrailStrategy
from backend.config import get_settings
from backend.state import AgentResult, GraphState
from backend.utils.log import get_logger
from backend.agents.sql_agent.sql_validator import validate_and_format_sql, SQLValidationError

logger = get_logger(__name__)

# Schema is hardcoded here for simplicity, but could be read dynamically via PRAGMA
DB_SCHEMA = """
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    manager_id INTEGER
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL,
    department_id INTEGER, -- references departments.id
    job_title TEXT NOT NULL,
    hire_date DATE NOT NULL,
    is_active BOOLEAN
);

CREATE TABLE leave_balances (
    employee_id INTEGER PRIMARY KEY, -- references employees.id
    annual_leave_days INTEGER,
    sick_leave_days INTEGER,
    maternity_paternity_leave_days INTEGER
);

CREATE TABLE salary_history (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL, -- references employees.id
    salary_amount DECIMAL(10,2) NOT NULL,
    effective_date DATE NOT NULL
);
"""

SQL_SYSTEM_PROMPT = """You are a highly skilled SQLite data analyst for an enterprise HR system.
Your job is to translate the user's natural language question into a syntactically correct SQLite query.

Here is the exact database schema you must use:
{schema}

Important Schema Context:
- The `employees` table links to `departments` via `department_id`.
- A department's manager is also an employee. `departments.manager_id` references `employees.id`.
- Always select human-readable columns (e.g., first_name and last_name) rather than just IDs when returning results.
- `leave_balances` and `salary_history` are linked to employees via `employee_id`.

Rules:
1. ONLY return the raw SQL query. Do not wrap it in markdown block quotes (e.g. no ```sql). 
2. Do not include any explanations, preambles, or postscripts. Just the raw SQL string.
3. You may ONLY use SELECT statements (no INSERT, UPDATE, DELETE).
4. The system automatically enforces a LIMIT clause; you do not need to add one.
5. Use proper JOINs, aggregations, and standard SQLite functions (e.g., date('now')) as needed.
6. When querying text, be robust against case-sensitivity where appropriate (e.g., use LOWER(name) = 'john doe').
"""


class SQLAgent(BaseAgent):
    """
    Agent that translates NL to SQL, validates it, and executes it.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        guardrails: list[GuardrailStrategy] | None = None,
    ) -> None:
        super().__init__(llm=llm, guardrails=guardrails)
        self.settings = get_settings()

    async def run(self, state: GraphState) -> AgentResult:
        """Execute the SQL generation and query pipeline."""
        t0 = time.monotonic()

        # 1. Extract query for this agent from the plan
        plan = state.get("plan")
        if not plan or "sql" not in plan.get("queries", {}):
            return AgentResult.failure(
                self.name,
                "No SQL query found in planner output.",
                self._timed_result(t0),
            )
            
        nl_query = plan["queries"]["sql"]
        logger.info("SQLAgent triggered", query=nl_query)

        # 2. Run guardrails
        guard_block = await self._run_with_guardrails(nl_query)
        if guard_block:
            return guard_block

        # 3. Generate SQL via LLM
        messages = [
            SystemMessage(content=SQL_SYSTEM_PROMPT.format(schema=DB_SCHEMA)),
            HumanMessage(content=nl_query),
        ]
        
        try:
            llm_response = await self._llm.ainvoke(messages)
            raw_sql = str(llm_response.content).strip()
            
            # Strip markdown block if the LLM hallucinated it despite instructions
            if raw_sql.startswith("```sql"):
                raw_sql = raw_sql[6:]
            if raw_sql.startswith("```"):
                raw_sql = raw_sql[3:]
            if raw_sql.endswith("```"):
                raw_sql = raw_sql[:-3]
            raw_sql = raw_sql.strip()

        except Exception as e:
            logger.error("LLM SQL generation failed", error=str(e))
            return AgentResult.failure(
                self.name, f"Failed to generate SQL: {e}", self._timed_result(t0)
            )

        # 4. Validate SQL (Security & Syntax)
        try:
            safe_sql = validate_and_format_sql(raw_sql, max_rows=self.settings.sql_max_rows)
            logger.info("SQL validated", original=raw_sql, safe=safe_sql)
        except SQLValidationError as e:
            logger.warning("SQL validation blocked query", original=raw_sql, error=str(e))
            return AgentResult.failure(
                self.name, f"Unsafe or invalid SQL generated: {e}", self._timed_result(t0)
            )

        # 5. Execute against SQLite
        try:
            async with aiosqlite.connect(self.settings.sqlite_db_path) as db:
                # Set row factory to return dicts
                db.row_factory = aiosqlite.Row
                async with db.execute(safe_sql) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description] if cursor.description else []
                    
                    # Convert aiosqlite.Row to plain dicts
                    row_dicts = [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error("SQLite execution error", error=str(e), sql=safe_sql)
            return AgentResult.failure(
                self.name, f"Database error during execution: {e}", self._timed_result(t0, {"sql": safe_sql})
            )

        # 6. Format the answer as a Markdown Table
        answer_md = self._format_as_markdown_table(columns, row_dicts, safe_sql)

        # 7. Return AgentResult
        meta = self._timed_result(t0, {"sql": safe_sql, "row_count": len(row_dicts)})
        
        return AgentResult(
            agent_name=self.name,
            success=True,
            answer=answer_md,
            structured_data={
                "sql_query": safe_sql,
                "columns": columns,
                "rows": row_dicts,
            },
            metadata=meta,
        )

    def _format_as_markdown_table(self, columns: list[str], rows: list[dict], sql_query: str) -> str:
        """Helper to build a markdown table from the query results."""
        if not columns or not rows:
            return f"**SQL Query executed:**\n```sql\n{sql_query}\n```\n\n*No results found.*"
            
        # Header
        header = "| " + " | ".join(columns) + " |"
        separator = "|" + "|".join(["---"] * len(columns)) + "|"
        
        # Rows
        row_lines = []
        for row in rows:
            line = "| " + " | ".join(str(row[col]) for col in columns) + " |"
            row_lines.append(line)
            
        table_str = "\n".join([header, separator] + row_lines)
        
        return f"**SQL Query executed:**\n```sql\n{sql_query}\n```\n\n**Results:**\n{table_str}"
