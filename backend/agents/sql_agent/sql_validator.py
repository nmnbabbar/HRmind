"""
backend/agents/sql_agent/sql_validator.py
=========================================
Security layer for the SQL Agent.

Validates that LLM-generated SQL queries are:
1. Syntactically valid in SQLite.
2. Contains exactly ONE statement.
3. Purely read-only (SELECT). No INSERT, UPDATE, DELETE, DROP, etc.
4. Bounded by a LIMIT clause to prevent massive data dumps.
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from backend.config import get_settings


class SQLValidationError(Exception):
    """Raised when a SQL query fails security or syntax validation."""
    pass


def validate_and_format_sql(query: str, max_rows: int | None = None) -> str:
    """
    Parses, validates, and formats a SQL query.
    Enforces read-only SELECT constraints.
    Injects or bounds the LIMIT clause to max_rows.

    Parameters
    ----------
    query : str
        The raw SQL query string from the LLM.
    max_rows : int | None
        The maximum number of rows allowed. Defaults to settings.sql_max_rows.

    Returns
    -------
    str
        The formatted, safe SQLite query string.

    Raises
    ------
    SQLValidationError
        If the query is invalid or violates security constraints.
    """
    if max_rows is None:
        max_rows = get_settings().sql_max_rows

    try:
        # Parse into a list of expressions (one per statement)
        statements = sqlglot.parse(query, read="sqlite")
    except ParseError as e:
        raise SQLValidationError(f"Invalid SQL syntax: {e}")

    # Remove None values (can happen with trailing semicolons or empty queries)
    statements = [stmt for stmt in statements if stmt is not None]

    if len(statements) == 0:
        raise SQLValidationError("No SQL statement found.")
        
    if len(statements) > 1:
        raise SQLValidationError("Multiple SQL statements are not allowed. Please provide a single query.")

    root = statements[0]

    # Ensure the root statement is a SELECT
    if not isinstance(root, exp.Select):
        raise SQLValidationError(f"Only SELECT statements are allowed. Found: {root.__class__.__name__}")

    # Walk the AST to ensure no mutating statements exist in subqueries or CTEs
    forbidden_classes = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, 
        exp.Alter, exp.Command, exp.Commit, exp.Rollback, exp.Pragma
    )
    for cls in forbidden_classes:
        if list(root.find_all(cls)):
            raise SQLValidationError(f"Forbidden statement found in query: {cls.__name__}")

    # Check and bound LIMIT
    limit = root.args.get("limit")
    if limit is not None:
        try:
            # sqlglot Limit expressions have an expression attribute for the value
            # Extract the literal integer value
            if isinstance(limit.expression, exp.Literal):
                limit_val = int(limit.expression.name)
                if limit_val > max_rows:
                    root.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
            else:
                # Dynamic limit (e.g., LIMIT ?), we safely overwrite it
                root.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        except (ValueError, AttributeError, TypeError):
            root.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        # No limit clause exists, so add one
        root.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    # Return the clean, formatted SQL
    return root.sql(dialect="sqlite")
