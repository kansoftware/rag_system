from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import settings

# Создаем "движок" SQLAlchemy
# pool_pre_ping=True - проверяет соединение перед использованием
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
        yield db
    finally:
        db.close()