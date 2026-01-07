from fastapi import FastAPI
from .routes import router as api_router
from .llm import get_llm_client, close_llm_client
from src.ingestion.embedding import get_embedding_model
from .reranker import get_reranker_model


app = FastAPI(
    title="RAG System API",
    description="API для вопросно-ответной системы с использованием RAG.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    """
    Инициализирует ML-модели при старте приложения.
    """
    print("Application startup: Initializing models...")
    # Инициализация происходит через get_... функции, которые кэшируются
    get_embedding_model()
    get_reranker_model()
    get_llm_client()
    print("Models initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Корректно закрывает соединения при остановке.
    """
    print("Application shutdown: Closing resources...")
    await close_llm_client()
    print("Resources closed.")

app.include_router(api_router)

@app.get("/", tags=["Health Check"])
def health_check():
    """
    Эндпоинт для проверки работоспособности сервиса.
    """
    return {"status": "ok"}