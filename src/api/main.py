import logging

from fastapi import FastAPI

from src.api.llm import close_llm_client, get_llm_client
from src.api.reranker import get_reranker_model
from src.api.routes import router as api_router
from src.ingestion.embedding import get_embedding_model
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="API для вопросно-ответной системы с использованием RAG.",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    """
    Инициализирует ML-модели при старте приложения.
    """
    setup_logging()
    logger.info("Application startup: Initializing models...")
    # Инициализация происходит через get_... функции, которые кэшируются
    get_embedding_model()
    get_reranker_model()
    get_llm_client()
    logger.info("Models initialized.")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Корректно закрывает соединения при остановке.
    """
    logger.info("Application shutdown: Closing resources...")
    await close_llm_client()
    logger.info("Resources closed.")


app.include_router(api_router)


@app.get("/", tags=["Health Check"])
def health_check():
    """
    Эндпоинт для проверки работоспособности сервиса.
    """
    return {"status": "ok"}
