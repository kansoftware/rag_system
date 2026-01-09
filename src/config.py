from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    # Конфигурация модели Pydantic для чтения из .env файла
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "rag_docs"
    POSTGRES_USER: str = "rag_user"
    POSTGRES_PASSWORD: str = ""

    # LLM
    LLM_PROVIDER: str = "lmstudio"
    LLM_BASE_URL: str = "http://localhost:1234/v1"
    LLM_API_KEY: str = "lm-studio"
    LLM_MODEL: str = "local-model"
    LLM_TIMEOUT: int = 90

    # Embedding
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Reranker
    RERANKER_MODEL: str = "sentence-transformers/ms-marco-MiniLM-L-12-v2"
    RERANKER_DEVICE: str = "cpu"
    RERANKER_BATCH_SIZE: int = 16

    # RAG
    TOP_K_INITIAL: int = 30
    TOP_K_FINAL: int = 7
    MIN_CONFIDENCE: float = 0.7
    ENABLE_RERANKER: bool = True
    
    # Chunking
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100

    # Django
    DJANGO_SECRET_KEY: str = ""
    DJANGO_DEBUG: bool = True
    DJANGO_ALLOWED_HOSTS: str = "localhost,127.0.0.1"

    # FastAPI
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

# Создаем единственный экземпляр настроек, который будет использоваться во всем приложении
settings = Settings()