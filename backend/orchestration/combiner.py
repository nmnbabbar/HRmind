import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.state import GraphState, AgentResult
from backend.config import get_settings

logger = logging.getLogger(__name__)

COMBINER_SYSTEM_PROMPT = """You are the final synthesizer for an HR AI assistant.
Your job is to read the results from various specialized agents (RAG, SQL, DocParser) and generate a cohesive, natural, and helpful response to the user.

Rules:
1. Synthesize all the agent answers into a single smooth reply.
2. Do NOT mention the names of the internal agents (e.g., don't say "The SQL agent found..." or "The RAG agent says..."). Just say "I found..." or provide the answer.
3. If any agent reported partial or failed results, include a polite caveat or warning.
4. Keep the tone professional, helpful, and concise.

Agent Results:
{agent_results}

User Query:
{query}
"""

def combiner_node(state: GraphState) -> GraphState:
    """
    Combiner Node: Reads agent_results and synthesizes a final answer.
    """
    agent_results = state.get("agent_results", [])
    
    if not agent_results:
        return {"final_answer": "I'm sorry, I couldn't process your request. No agents returned results."}
        
    formatted_results = ""
    warnings = []
    
    for res_dict in agent_results:
        res = AgentResult.from_dict(res_dict)
        if res.success:
            formatted_results += f"- {res.answer}\n"
            # Check completeness score from doc_parser
            comp_score = res.metadata.get("completeness_score")
            if comp_score is not None and comp_score < 0.4:
                warnings.append("⚠️ Only partial information was extracted from the document. Please cross-check the values.")
        else:
            formatted_results += f"- Error processing request: {res.error}\n"

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.combiner_model, 
        temperature=0.2,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", COMBINER_SYSTEM_PROMPT),
        ("user", "Please provide the final response.")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "agent_results": formatted_results,
        "query": state["query"]
    })
    
    final_text = response.content
    
    if warnings:
        # Append unique warnings at the end
        unique_warnings = list(dict.fromkeys(warnings))
        final_text += "\n\n" + "\n".join(unique_warnings)
        
    return {"final_answer": str(final_text)}
