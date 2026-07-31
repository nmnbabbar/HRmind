from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

from backend.state import GraphState, PlannerOutput

PLANNER_SYSTEM_PROMPT = """You are the master routing planner for an enterprise HR AI assistant.
Your responsibility is to analyze the user's query and meticulously decide which specialized agents to invoke to construct a complete, accurate answer.

Available agents:
- "rag": Semantic search over HR policy documents, handbooks, and general company guidelines. Select this for questions about rules, policies, benefits, processes, or general company knowledge.
- "sql": Database querying for structured HR and employee data. Select this for queries requiring numerical data, specific employee lookups, payroll details, department stats, or leave balances.
- "doc_parser": OCR and text extraction from uploaded files (e.g., contracts, payslips, ID cards).

Execution Rules:
1. File Uploads: If `uploaded_file_path` is provided, ALWAYS include "doc_parser".
2. Parsed Context: If `has_parsed_doc` is true BUT `uploaded_file_path` is None, the file was already parsed. DO NOT include "doc_parser". Rely on the `entity_store` for extracted values.
3. Policy/Rules: If the query asks about how something works, company rules, or policies, ALWAYS include "rag".
4. Data/Stats: If the query asks for specific employee facts, numbers, or records, ALWAYS include "sql".
5. Conversational/Greetings: If the user's query is just a conversational greeting (like "hi", "hello", "thanks") or off-topic, set `agents` to an empty list `[]`. Do NOT invoke "sql" or "rag" for greetings.
6. Execution Order (`parallel`): 
   - Set `parallel=True` if the selected agents can run independently.
   - Set `parallel=False` if one agent explicitly depends on the output of another (e.g., DocParser must extract an employee ID before SQL can query their record).

Output: Provide the selected agents, the `parallel` flag, and a specific rewritten `query` for EACH selected agent. The rewritten query should give that agent clear, specialized instructions (e.g., for SQL, specify exactly what data is needed; for RAG, specify the exact policy topic).
"""

from backend.config import get_settings

def planner_node(state: GraphState) -> GraphState:
    """
    Planner Node: Analyzes the state and determines the execution plan.
    Returns a dict with the `plan` key containing the serialized PlannerOutput.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.planner_model, 
        temperature=0,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key
    ).with_structured_output(PlannerOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("user", "User Query: {query}\n\nUploaded File Path: {uploaded_file_path}\nParsed Document exists: {has_parsed_doc}\nEntity Store: {entity_store}")
    ])
    
    chain = prompt | llm
    
    has_parsed_doc = state.get("parsed_document") is not None
    
    plan: PlannerOutput = chain.invoke({
        "query": state["query"],
        "uploaded_file_path": state.get("uploaded_file_path"),
        "has_parsed_doc": has_parsed_doc,
        "entity_store": state.get("entity_store", {})
    })
    
    return {"plan": plan.to_dict()}
