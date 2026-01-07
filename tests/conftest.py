import sys
import os

# Добавляем корневую директорию проекта в sys.path ПЕРЕД всеми остальными импортами
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.models import Base
from src.db.session import get_db
from src.api.main import app

# --- Настройка тестовой базы данных SQLite in-memory ---

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def test_db():
    """
    Фикстура для создания и очистки тестовой базы данных для каждого теста.
    """
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function", autouse=True)
def override_get_db(test_db):
    """
    Автоматически подменяет зависимость get_db на тестовую сессию.
    """
    def _override_get_db():
        try:
            yield test_db
        finally:
            test_db.close()

    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)