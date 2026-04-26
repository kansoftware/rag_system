import time
from typing import Annotated, Union

from fastapi import APIRouter, Depends, Query
from fastapi import Request as FastAPIRequest
from sqlalchemy.orm import Session

from src.db.session import get_db
from src.ingestion.embedding import get_embedding_model

from .dependencies import get_rag_engine
from .llm import get_llm_client
from .rag import RAGEngine
from .schemas import (
    FallbackResponse,
    LLMInfo,
    PaginatedHistoryResponse,
    QueryHistoryItem,
    QueryRequest,
    QueryResponse,
)
from .services.history_service import QueryHistoryService

router = APIRouter(prefix="/api/v1", tags=["RAG"])


@router.post("/query", response_model=Union[QueryResponse, FallbackResponse])
async def query_endpoint(
    request: QueryRequest,
    # TODO: Заменить на реальную аутентификацию
    fastapi_request: FastAPIRequest,
    db: Annotated[Session, Depends(get_db)],
    rag_engine: Annotated[RAGEngine, Depends(get_rag_engine)],
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
        temperature=request.temperature,
    )

    llm_client = get_llm_client()
    query_id = QueryHistoryService.save(
        db, int(user_id), request.query, query_embedding, result, llm_client
    )

    if "fallback" in result.get("warnings", []):
        return FallbackResponse(
            query_id=query_id,
            response_md=result["response_md"],
            confidence_score=result.get("confidence_score", 0.0),
            sources=result.get("sources", []),
            warnings=result.get("warnings", []),
        )

    return QueryResponse(
        query_id=query_id,
        response_md=result["response_md"],
        confidence_score=result["confidence_score"],
        sources=result["sources"],
        llm=LLMInfo(provider=llm_client.provider, model=llm_client.model),
        timings_ms=result["timings_ms"],
        warnings=result.get("warnings", []),
    )

    # except Exception as e:
    #     print(f"An error occurred during query processing: {e}")
    #     db.rollback()
    #     raise HTTPException(
    #         status_code=500,
    #         detail="An internal error occurred while processing the request."
    #     ) from e


@router.get("/history", response_model=PaginatedHistoryResponse)
async def get_query_history(
    db: Annotated[Session, Depends(get_db)],
    user_id: int = Query(..., description="ID пользователя"),
    page: int = Query(1, ge=1, description="Номер страницы (начиная с 1)"),
    limit: int = Query(
        10, ge=1, le=100, description="Количество записей на странице"
    ),
):
    """
    Получает историю запросов пользователя с пагинацией.
    """
    items, total = QueryHistoryService.get_user_history(db, user_id, page, limit)
    # Преобразуем ORM-объекты в Pydantic модели с помощью from_orm (ORM mode)
    items_models = [QueryHistoryItem.from_orm(hist) for hist in items]
    return PaginatedHistoryResponse(total=total, page=page, limit=limit, items=items_models)
