from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import src.api.routes as routes_module

# Важно: импортируем `app` и зависимости до того, как моки их заменят
from src.api.dependencies import get_rag_engine
from src.api.main import app
from src.api.rag import RAGEngine

# --- Моки для ML моделей и зависимостей ---

@pytest.fixture(scope="module")
def mock_rag_engine():
    """Мок RAGEngine, который возвращает предопределенные данные."""
    engine = MagicMock(spec=RAGEngine)
    
    async def mock_query(*args, **kwargs):
        return {
            "query_id": "test_query_123",
            "response_md": "Это тестовый ответ от LLM.",
            "confidence_score": 0.95,
            "sources": [
                {
                    "source_id": 1,
                    "chunk_id": 101,
                    "document_id": 1,
                    "title": "Тестовый документ",
                    "url": "http://example.com/test",
                    "path": "/data/test.md",
                    "domain": "example.com",
                    "similarity": 0.88,
                    "rerank_score": 0.92,
                    "excerpt": "Фрагмент тестового документа..."
                }
            ],
            "llm": {"provider": "mock", "model": "mock-model"},
            "timings_ms": {
                "embed": 10,
                "retrieve": 20,
                "rerank": 30,
                "llm": 90,
                "total": 150
            },
            "warnings": []
        }
        
    engine.query = AsyncMock(side_effect=mock_query)
    return engine

@pytest_asyncio.fixture(scope="function", autouse=True)
def override_dependencies(mock_rag_engine):
    """Переопределяем зависимость get_rag_engine для всех тестов в этом модуле."""
    app.dependency_overrides[get_rag_engine] = lambda: mock_rag_engine
    yield
    app.dependency_overrides = {}

# --- Тесты ---

@pytest.mark.asyncio
async def test_query_endpoint_success(mocker):
    """
    Проверяет успешный ответ от эндпоинта /api/v1/query.
    """
    mocker.patch.object(routes_module, 'save_query_history', return_value=999)
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"query": "тестовый запрос"})

    assert response.status_code == 200
    data = response.json()
    
    assert data["query_id"] == 999
    assert "Это тестовый ответ" in data["response_md"]
    assert data["confidence_score"] == 0.95
    assert len(data["sources"]) == 1
    assert data["sources"][0]["title"] == "Тестовый документ"

@pytest.mark.asyncio
async def test_query_endpoint_empty_query(mocker):
    """
    Проверяет, что пустой запрос возвращает ошибку 422.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"query": ""})
    
    assert response.status_code == 422 # Unprocessable Entity
    data = response.json()
    assert "detail" in data
    assert "query" in data["detail"][0]["loc"]

@pytest.mark.asyncio
async def test_query_endpoint_invalid_body(mocker):
    """
    Проверяет, что запрос с некорректным телом (без поля query) возвращает 422.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"text": "неправильное поле"})
        
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"][0]["msg"] == "Field required"
    assert "query" in data["detail"][0]["loc"]

@pytest.mark.asyncio
async def test_rag_engine_called_with_params(mock_rag_engine, mocker):
    """
    Проверяет, что RAGEngine.query вызывается с правильными параметрами из запроса.
    """
    mock_rag_engine.query.reset_mock() # Сбрасываем мок перед тестом
    
    query_params = {
        "query": "специальный запрос",
        "top_k_final": 7,
        "domain_filter": "docs.python.org"
    }
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/api/v1/query", json=query_params)
        
    # Проверяем, что мок был вызван один раз с ожидаемыми аргументами
    mock_rag_engine.query.assert_called_once()
    call_args, call_kwargs = mock_rag_engine.query.call_args
    
    # Все параметры передаются как именованные
    assert call_kwargs.get("query_text") == "специальный запрос"
    assert call_kwargs.get("top_k_final") == 7
    # Проверяем, что domain_filter не передается в query, так как его там нет
    assert "domain_filter" not in call_kwargs
