from fastapi import Depends
from sqlalchemy.orm import Session
from functools import lru_cache

from .llm import get_llm_client
from src.ingestion.embedding import get_embedding_model
from .reranker import get_reranker_model
from .rag import RAGEngine
from src.db.session import get_db

@lru_cache(maxsize=1)
def get_rag_engine() -> RAGEngine:
    """
    Зависимость для получения единственного экземпляра RAGEngine.
    Использует lru_cache для создания синглтона.
    """
    return RAGEngine(
        embedding_model=get_embedding_model(),
        reranker_model=get_reranker_model(),
        llm_client=get_llm_client()
    )