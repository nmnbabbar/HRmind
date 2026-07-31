import logging
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.agents.rag_agent.rag_agent import RAGAgent
from backend.agents.sql_agent.sql_agent import SQLAgent
from backend.agents.doc_parser_agent.doc_parser_agent import DocParserAgent

# RAG dependencies will be imported lazily inside get_agent to avoid
# circular imports and missing modules if they are initialized elsewhere.
from backend.agents.rag_agent.guardrails import TopicGuardrail, GroundingGuardrail

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Factory to instantiate agents with their dependencies.
    Caches instances to avoid rebuilding vector connections/indexes repeatedly.
    """
    _agents = {}
    
    @classmethod
    def get_agent(cls, agent_name: str):
        if agent_name in cls._agents:
            return cls._agents[agent_name]
            
        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.agent_model, 
            temperature=0,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key
        )
        
        if agent_name == "sql":
            agent = SQLAgent(llm=llm)
        elif agent_name == "doc_parser":
            agent = DocParserAgent(llm=llm)
        elif agent_name == "rag":
            # Instantiate RAG dependencies (mocked or lazily imported for now)
            # In a real setup, these dependencies are injected by a DI container or initialized at startup.
            guardrails = [TopicGuardrail(llm=llm), GroundingGuardrail(llm=llm)]
            
            agent = RAGAgent(
                llm=llm,
                embedding_service=None, # type: ignore
                vector_repo=None,       # type: ignore
                bm25_index=None,        # type: ignore
                settings=settings,
                guardrails=guardrails
            )
        else:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        cls._agents[agent_name] = agent
        return agent
