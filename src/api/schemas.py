from typing import List, Optional

from pydantic import BaseModel, Field
from src.config import settings


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Текст запроса пользователя")
    top_k_initial: int = Field(settings.TOP_K_INITIAL, ge=1, le=100, description="Количество кандидатов для αρχικού поиска")
    top_k_final: int = Field(settings.TOP_K_FINAL, ge=1, le=20, description="Количество финальных источников после ре-ранжирования")
    domain_filter: Optional[str] = Field(None, description="Фильтр по домену источников")
    min_confidence: float = Field(settings.MIN_CONFIDENCE, ge=0.0, le=1.0, description="Минимальный порог уверенности для ответа")
    temperature: float = Field(0.3, ge=0.0, le=1.0, description="Температура для генерации LLM")

class Source(BaseModel):
    source_id: int = Field(..., description="Порядковый номер источника в ответе")
    chunk_id: int
    document_id: int
    title: Optional[str]
    url: Optional[str] = None
    path: Optional[str] = None
    domain: Optional[str] = None
    similarity: Optional[float] = Field(None, description="Сходство от векторного поиска")
    rerank_score: Optional[float] = Field(None, description="Оценка релевантности от ре-ранкера")
    excerpt: str = Field(..., description="Краткий фрагмент текста чанка")

class LLMInfo(BaseModel):
    provider: str
    model: str

class Timings(BaseModel):
    embed: float
    retrieve: float
    rerank: float
    llm: float
    total: float

class QueryResponse(BaseModel):
    query_id: int
    response_md: str = Field(..., description="Ответ от LLM в формате Markdown")
    confidence_score: float
    sources: List[Source]
    llm: LLMInfo
    timings_ms: Timings
    warnings: List[str] = []

class FallbackResponse(BaseModel):
    query_id: int
    response_md: str = Field("К сожалению, недостаточно информации для уверенного ответа.", description="Fallback-сообщение")
    confidence_score: float
    sources: List[Source]
    warnings: List[str]