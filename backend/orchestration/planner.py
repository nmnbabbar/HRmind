from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

from backend.state import GraphState, PlannerOutput

PLANNER_SYSTEM_PROMPT = """You are a routing planner for an HR AI assistant.
Your job is to analyze the user's query and decide which agents to invoke to construct a complete answer.

Available agents:
- "rag": Searches HR policy documents and general company information. Use for questions about policies, benefits, guidelines.
- "sql": Queries the employee and HR database. Use for questions requiring data, numbers, employee lookups, payroll, leave balances.
- "doc_parser": Extracts structured information from uploaded files (contracts, payslips, ID cards).

Rules:
1. If no file is uploaded (uploaded_file_path is None) AND parsed_document is None, NEVER include "doc_parser".
2. If a file is uploaded (uploaded_file_path is provided), ALWAYS include "doc_parser".
3. If parsed_document is provided BUT uploaded_file_path is None, this means a file was already parsed in a previous turn. Use the entity_store for context and DO NOT include "doc_parser".
4. If the user asks a question about policies, include "rag".
5. If the user asks about specific employee data, include "sql".
6. Set `parallel=True` if the agents are independent. Set `parallel=False` if one agent depends on the output of another (e.g., if you need DocParser to extract an employee ID before SQL can query their leave balance, parallel is False).

Output queries for each agent you select. The query should give the agent clear instructions based on the user's request.
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
