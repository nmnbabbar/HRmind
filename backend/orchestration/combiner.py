import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.state import GraphState, AgentResult
from backend.config import get_settings

logger = logging.getLogger(__name__)

COMBINER_SYSTEM_PROMPT = """You are the lead HR AI assistant, acting as the primary conversational interface for employees.
Your job is to read the raw data and results provided by specialized background agents (RAG, SQL, DocParser) and synthesize them into a cohesive, natural, and highly professional response for the user.

Rules for Synthesis:
1. Direct Communication: Do NOT mention the internal agents (e.g., avoid "The SQL agent found..." or "According to the RAG system..."). Speak as a unified AI assistant (e.g., "I found that...", "Our records indicate...").
2. Conversational Context: If `Agent Results` indicates that "No specialized agents were invoked", it means the user simply greeted you or made a conversational remark. Respond warmly and conversationally without apologizing for missing data.
3. Clarity and Formatting: Present the information clearly. Use markdown formatting (bolding, bullet points) when listing multiple data points or policies to make it readable.
4. Handling Failures: If any agent reported partial or failed results (e.g., an error message is present), include a polite, brief caveat or apology for that specific piece of missing information.
5. Tone: Maintain a professional, empathetic, and extremely helpful tone suitable for enterprise HR. Do not hallucinate or invent policies or data.

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
    
    formatted_results = ""
    warnings = []
    
    if not agent_results:
        formatted_results = "No specialized agents were invoked. Just respond conversationally to the user."
    else:
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
