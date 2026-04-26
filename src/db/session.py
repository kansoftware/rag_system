from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import settings

# Создаем "движок" SQLAlchemy
# pool_pre_ping=True - проверяем соединение перед использованием
# pool_recycle=3600 - пересоздает соединение каждый час
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)

# Создаем фабрику сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Зависимость FastAPI для получения сессии БД.
    Гарантирует, что сессия будет закрыта после завершения запроса.
    """
    db = SessionLocal()
    try:
        # Валидация соответствия размерности эмбеддингов модели и колонки БД
        from src.db.models import QueryHistory
        from src.ingestion.embedding import get_embedding_model

        embedding_model = get_embedding_model()
        # Получаем Column объект для query_embedding
        query_embedding_column = QueryHistory.__table__.c.get("query_embedding")
        if query_embedding_column is not None:
            db_dim = getattr(query_embedding_column.type, "dim", None)
            if db_dim is not None and embedding_model.embedding_dim != db_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: model has {embedding_model.embedding_dim}, "
                    f"but DB column expects {db_dim}. "
                    "Run migrations to update the vector column dimension or change the embedding model."
                )
        yield db
    finally:
        db.close()
