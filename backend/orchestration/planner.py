from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
import os
import logging
from better_profanity import profanity

from backend.state import GraphState, PlannerOutput

logger = logging.getLogger(__name__)
MAX_QUERY_LENGTH = 1000

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
5. Conversational/Out of Scope: If the user's query is a conversational greeting (like "hi", "hello") OR falls outside the scope of HR, company policies, or employee data (e.g. asking to write code, general trivia), set `agents` to an empty list `[]`. In the `reasoning` field, provide a direct, polite fallback response to the user (e.g., "Hi! I am the HR AI assistant. How can I help you?" or "I'm sorry, I can only assist with HR-related inquiries.").
6. Execution Order (`parallel`): 
   - Set `parallel=True` if the selected agents can run independently.
   - Set `parallel=False` if one agent explicitly depends on the output of another (e.g., DocParser must extract an employee ID before SQL can query their record).
7. Output: Provide the selected agents, the `parallel` flag, and a specific rewritten `query` for EACH selected agent. The rewritten query should give that agent clear, specialized instructions.
8. Context Resolution: You may receive the "Previous Turn" context. Use this ONLY to resolve pronouns or missing context in the current User Query (e.g. if the user says "what are its aims", refer to the previous turn to know what "it" is, and rewrite the query for the agent as "what are the aims of the alcohol policy"). Do NOT answer the question yourself.
"""

from backend.config import get_settings

def planner_node(state: GraphState) -> GraphState:
    """
    Planner Node: Analyzes the state and determines the execution plan.
    Returns a dict with the `plan` key containing the serialized PlannerOutput.
    """
    query = state.get("query", "")
    
    # 1. Input Length Guardrail
    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(f"Query length {len(query)} exceeds MAX_QUERY_LENGTH {MAX_QUERY_LENGTH}")
        ans = f"Your query is too long ({len(query)} characters). Please keep it under {MAX_QUERY_LENGTH} characters."
        return {
            "plan": PlannerOutput(agents=[], parallel=False, queries={}, reasoning=ans).to_dict(),
            "final_answer": ans,
            "previous_turn": {"query": query, "answer": ans}
        }
        
    # 2. Profanity Guardrail
    if profanity.contains_profanity(query):
        logger.warning("Profanity detected in user query.")
        ans = "Please maintain a professional tone. I cannot process queries containing profanity."
        return {
            "plan": PlannerOutput(agents=[], parallel=False, queries={}, reasoning=ans).to_dict(),
            "final_answer": ans,
            "previous_turn": {"query": query, "answer": ans}
        }

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.planner_model, 
        temperature=0,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key
    ).with_structured_output(PlannerOutput)
    
    previous_turn = state.get("previous_turn")
    context_str = ""
    if previous_turn:
        context_str = f"\n\nPrevious Question: {previous_turn['query']}\nPrevious Answer: {previous_turn['answer']}"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("user", "User Query: {query}\n\nUploaded File Path: {uploaded_file_path}\nParsed Document exists: {has_parsed_doc}\nEntity Store: {entity_store}{context_str}")
    ])
    
    chain = prompt | llm
    
    has_parsed_doc = state.get("parsed_document") is not None
    
    try:
        plan: PlannerOutput = chain.invoke({
            "query": state["query"],
            "uploaded_file_path": state.get("uploaded_file_path"),
            "has_parsed_doc": has_parsed_doc,
            "entity_store": state.get("entity_store", {}),
            "context_str": context_str
        })
    except Exception as e:
        logger.error(f"JSON Schema Validation Error in Planner: {e}")
        ans = "I encountered an internal error while parsing the request. Please try again or rephrase."
        plan = PlannerOutput(agents=[], parallel=False, queries={}, reasoning=ans)
    
    update = {"plan": plan.to_dict()}
    
    # If no agents are needed, set the final_answer immediately so the graph can exit.
    if not plan.agents:
        final_ans = plan.reasoning or "I'm sorry, I cannot assist with that request."
        update["final_answer"] = final_ans
        update["previous_turn"] = {"query": state["query"], "answer": final_ans}
        
    return update
