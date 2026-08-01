import logging
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.agents.rag_agent.rag_agent import RAGAgent
from backend.agents.sql_agent.sql_agent import SQLAgent
from backend.agents.doc_parser_agent.doc_parser_agent import DocParserAgent



logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Factory to instantiate agents with their dependencies.
    Caches instances to avoid rebuilding vector connections/indexes repeatedly.
    """
    _agents = {}
    _embedding_service = None
    _vector_repo = None
    _bm25_index = None
    _is_initialized = False

    @classmethod
    async def initialize(cls):
        """Initialize heavy dependencies asynchronously (e.g. at startup)."""
        if cls._is_initialized:
            return
            
        from backend.agents.rag_agent.ingestion import EmbeddingService
        from backend.agents.rag_agent.retriever import build_bm25_index, create_chroma_repository
        from backend.config import get_settings
        
        settings = get_settings()
        logger.info("Initializing RAG dependencies...")
        
        cls._embedding_service = EmbeddingService(
            model_name="BAAI/bge-large-en-v1.5", 
        )
        cls._vector_repo = await create_chroma_repository()
        cls._bm25_index = await build_bm25_index(cls._vector_repo)
        
        cls._is_initialized = True
        logger.info("RAG dependencies initialized.")
    
    @classmethod
    def get_agent(cls, agent_name: str):
        if agent_name in cls._agents:
            return cls._agents[agent_name]
            
        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.agent_model, 
            temperature=0,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            max_retries=3
        )
        
        if agent_name == "sql":
            agent = SQLAgent(llm=llm)
        elif agent_name == "doc_parser":
            agent = DocParserAgent(llm=llm)
        elif agent_name == "rag":
            if not cls._is_initialized:
                logger.warning("AgentFactory not initialized! RAG dependencies might be missing.")
            
            agent = RAGAgent(
                llm=llm,
                embedding_service=cls._embedding_service,
                vector_repo=cls._vector_repo,
                bm25_index=cls._bm25_index,
                settings=settings
            )
        else:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        cls._agents[agent_name] = agent
        return agent
