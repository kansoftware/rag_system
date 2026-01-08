import json
import time
from typing import Annotated, Union, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Request as FastAPIRequest
from sqlalchemy.orm import Session

from src.db.models import QueryHistory
from src.db.session import get_db
from src.ingestion.embedding import get_embedding_model

from .dependencies import get_rag_engine
from .llm import get_llm_client
from .rag import RAGEngine
from .schemas import FallbackResponse, LLMInfo, QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1", tags=["RAG"])

def save_query_history(
    db: Session,
    user_id: int,
    query_text: str,
    query_embedding: list,
    result: dict,
    llm_client
) -> int:
    """Сохраняет результат запроса в базу данных."""
    history_entry = QueryHistory(
        user_id=user_id,
        query_text=query_text,
        query_embedding=query_embedding,
        response_md=result["response_md"],
        sources_json=json.loads(json.dumps([s for s in result.get("sources", [])], default=str)),
        llm_provider=llm_client.provider,
        llm_model=llm_client.model,
        confidence_score=result.get("confidence_score", 0.0),
    )
    db.add(history_entry)
    db.commit()
    db.refresh(history_entry)
    return cast(int, history_entry.id)


@router.post("/query", response_model=Union[QueryResponse, FallbackResponse])
async def query_endpoint(
    request: QueryRequest,
    # TODO: Заменить на реальную аутентификацию
    fastapi_request: FastAPIRequest,
    db: Annotated[Session, Depends(get_db)],
    rag_engine: Annotated[RAGEngine, Depends(get_rag_engine)]
):
    """
    Основной эндпоинт для выполнения RAG-запросов.
    """
    # try:
    # ЗАГЛУШКА: Получаем ID пользователя. В реальном приложении это будет из токена.
    user_id = fastapi_request.headers.get("X-User-Id", "1")

    emb_model = get_embedding_model()
    
    embed_start_time = time.time()
    query_embedding = emb_model.get_embeddings([request.query])[0]
    embed_time = (time.time() - embed_start_time) * 1000
    
    result = await rag_engine.query(
        db=db,
        query_text=request.query,
        query_embedding=query_embedding,
        embed_time_ms=embed_time,
        top_k_initial=request.top_k_initial,
        top_k_final=request.top_k_final,
        min_confidence=request.min_confidence,
        temperature=request.temperature
    )

    llm_client = get_llm_client()
    query_id = save_query_history(db, int(user_id), request.query, query_embedding, result, llm_client)
    
    if "fallback" in result.get("warnings", []):
            return FallbackResponse(
            query_id=query_id,
            response_md=result["response_md"],
            confidence_score=result.get("confidence_score", 0.0),
            sources=result.get("sources", []),
            warnings=result.get("warnings", [])
            )
    
    return QueryResponse(
        query_id=query_id,
        response_md=result["response_md"],
        confidence_score=result["confidence_score"],
        sources=result["sources"],
        llm=LLMInfo(provider=llm_client.provider, model=llm_client.model),
        timings_ms=result["timings_ms"],
        warnings=result.get("warnings", [])
    )

    # except Exception as e:
    #     print(f"An error occurred during query processing: {e}")
    #     db.rollback()
    #     raise HTTPException(
    #         status_code=500,
    #         detail="An internal error occurred while processing the request."
    #     ) from e