import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, cast

from sqlalchemy import asc
from sqlalchemy.orm import Session
from transformers import AutoTokenizer

from src.config import settings
from src.db.models import Chunk, Document
from src.ingestion.embedding import EmbeddingModel

from .llm import LLMClient
from .reranker import RerankerModel

logger = logging.getLogger(__name__)

# Fallback prompt template (встроенный, используется если файл не найден)
DEFAULT_PROMPT = """Вы — первоклассный помощник по работе с технической документации, специалист широкого профиля в computer science.
Ваша задача — отвечать на вопросы пользователей, основываясь ТОЛЬКО на предоставленной информации.

**// СТРОГИЕ ПРАВИЛА //**
1. **НИКОГДА НЕ ИЗОБРЕТАЙТЕ ИНФОРМАЦИЮ.** Каждое утверждение должно основываться на ПРЕДОСТАВЛЕННЫХ ИСТОЧНИКАХ.

2. **ССЫЛАЙТЕСЬ НА КАЖДОЕ ПРЕДЛОЖЕНИЕ.** Каждое написанное вами предложение должно заканчиваться ссылкой, например, `[SOURCE 1]` или `[SOURCE 1, 2]`.

3. **СИНТЕЗИРУЙТЕ, НЕ КОПИРУЙТЕ.** Объясняйте концепции своими словами. Если пользователь запрашивает пример кода, напишите понятный, работающий пример, основанный на информации из источников. Отвечайте только на русском языке. Все комментари в кода должны быть только на русском языке.

4. **ОБРАЩАЙТЕСЬ С ОТСУТСТВУЮЩЕЙ ИНФОРМАЦИЕЙ.** Только если источники не содержат никакой релевантной информации для ответа на вопрос, вы должны ответить одним предложением: «Информация не найдена в предоставленных источниках». Не добавляйте никакой другой текст.

5. Перед генерацией ответа убедись, что:
  - вам достаточно информации
  - что Вы понимаете вопрос
  - что ответ будет соответствовать правилам
  - что цитаты имеют правильный формат `[SOURCE 1]` или `[SOURCE 1, 2]`


**// ПРИМЕР ХОРОШЕГО ОТВЕТА //**
*ВОПРОС ПОЛЬЗОВАТЕЛЯ:* Как использовать boost::bimap?

*ВАШ ОТВЕТ:*
Boost.Bimap предоставляет двунаправленную карту, позволяющую осуществлять поиск по ключу или значению [SOURCE 2]. Для ее использования необходимо включить заголовочный файл `<boost/bimap.hpp>` [SOURCE 1].

Вот простой пример:
```cpp
#include <boost/bimap.hpp>
#include <iostream>
#include <string>

int main() {{
boost::bimap<int, std::string> bm;

bm.insert({{ 1, "one" }});

bm.insert({{ 2, "two" }});

// Поиск по ключу
std::cout << bm.left.at(1) << std::endl; // "one"

// Поиск по значению
std::cout << bm.right.at("two") << std::endl; // 2
}}
```
Этот пример демонстрирует создание bimap и доступ к его левому и правому представлениям для поиска [SOURCE 1, 3]. Более сложные примеры можно найти в документации Boost [SOURCE 4].

**// КОНЕЦ ПРИМЕРА //**

**// ЗАДАЧА ПОЛЬЗОВАТЕЛЯ //**
ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}

PROVIDED SOURCES:
---
{context}
---

ОТВЕТ (с указанием источников, с соблюдением всех правил и формата примера):"""


class RAGEngine:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        reranker_model: RerankerModel,
        llm_client: LLMClient,
    ):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.llm = llm_client
        self.tokenizer = AutoTokenizer.from_pretrained(
            reranker_model.model_name
        )  # Используем ту же модель, что и ре-ранкер/эмбеддер

        # Загрузка промпта из внешнего файла
        prompt_path = Path(__file__).parent / "prompts" / "rag_template.txt"
        try:
            self.prompt_template = prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("rag_template.txt not found, using built-in prompt template.")
            self.prompt_template = DEFAULT_PROMPT

    async def query(
        self,
        db: Session,
        query_text: str,
        query_embedding: List[float],
        embed_time_ms: float,
        top_k_initial: int,
        top_k_final: int,
        min_confidence: float,
        temperature: float,
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Эмбеддинг уже вычислен, замеряем только поиск
        candidates = self._vector_search(db, query_embedding, top_k_initial)
        retrieve_time = time.time() - start_time

        if not candidates:
            return self._generate_fallback_response(
                [], "No relevant documents found.", embed_time_ms=embed_time_ms
            )

        self._log_chunks(candidates, "Initial retrieval", "similarity")

        rerank_time_start = time.time()

        if settings.ENABLE_RERANKER:
            reranked_chunks = await asyncio.to_thread(
                self.reranker_model.rerank, query_text, candidates
            )
            self._log_chunks(reranked_chunks, "After Reranking", "rerank_score")
        else:
            logger.info("Reranking is disabled. Using similarity scores.")
            # Для совместимости с остальным кодом, который ожидает 'rerank_score'
            for chunk in candidates:
                chunk["rerank_score"] = chunk.get("similarity", 0.0)
            reranked_chunks = sorted(
                candidates, key=lambda x: x["rerank_score"], reverse=True
            )

        # Сначала фильтруем по порогу, потом берем топ
        confident_chunks = [
            chunk for chunk in reranked_chunks if chunk.get("rerank_score", 0.0) > 0.5
        ]
        final_chunks = confident_chunks[:top_k_final]
        rerank_time = time.time() - rerank_time_start

        prompt = self._build_prompt(query_text, final_chunks)

        llm_start_time = time.time()
        llm_response_text = await self.llm.generate(prompt, temperature)
        llm_time = time.time() - llm_start_time

        verified_sources = self._verify_citations(llm_response_text, final_chunks)
        confidence = self._calculate_confidence(verified_sources, llm_response_text)

        logger.debug("Confidence: %.2f (min_confidence: %s)", confidence, min_confidence)

        total_time = time.time() - start_time

        if confidence < min_confidence:
            return self._generate_fallback_response(
                chunks=final_chunks,
                warning=f"Confidence score {confidence:.2f} is below threshold {min_confidence}.",
                embed_time_ms=embed_time_ms,
                confidence=confidence,
                total_time=total_time,
            )

        return {
            "response_md": llm_response_text,
            "confidence_score": confidence,
            "sources": verified_sources,
            "timings_ms": {
                "embed": embed_time_ms,
                "retrieve": retrieve_time * 1000,
                "rerank": rerank_time * 1000,
                "llm": llm_time * 1000,
                "total": total_time * 1000,
            },
        }

    def _vector_search(
        self, db: Session, query_embedding: List[float], top_k: int
    ) -> List[Dict]:
        # Используем косинусное расстояние, так как индекс создан с vector_cosine_ops
        distance = (Chunk.embedding.cosine_distance(query_embedding)).label("distance")
        results = (
            db.query(Chunk, distance)
            .join(Document)
            .order_by(asc(distance))
            .limit(top_k)
            .all()
        )

        return [
            {
                "chunk_id": r.Chunk.id,
                "document_id": r.Chunk.document_id,
                "text": r.Chunk.chunk_text,
                "title": r.Chunk.document.title,
                "url": r.Chunk.document.source_url,
                "similarity": max(0.0, 1.0 - float(r.distance)),
            }
            for r in results
        ]

    def _build_prompt(self, query: str, chunks: List[Dict]) -> str:
        context = "\n\n---\n\n".join(
            [
                f"[SOURCE {i + 1}] (Title: {c.get('title')}, URL: {c.get('url')})\n{c['text']}"
                for i, c in enumerate(chunks)
            ]
        )

        prompt = self.prompt_template.format(query=query, context=context)

        # --- Detailed Logging for Prompt ---
        logger.debug("LLM Prompt: user_query='%s', chunks_count=%d", query, len(chunks))
        for i, chunk in enumerate(chunks):
            token_count = len(self.tokenizer.encode(chunk["text"]))
            logger.debug(
                "  - Chunk %d (rerank_score=%.4f, tokens=%d)",
                i + 1,
                chunk.get("rerank_score", 0.0),
                token_count,
            )
        logger.debug(
            "Total tokens in context: %d", len(self.tokenizer.encode(context))
        )
        return prompt

    def _verify_citations(self, response_text: str, chunks: List[Dict]) -> List[Dict]:
        cited_indices = set()

        # Ищем одиночные цитаты: [SOURCE 1]
        single_citations = re.finditer(r"\[SOURCE (\d+)\]", response_text)
        for m in single_citations:
            cited_indices.add(int(m.group(1)))

        # Ищем списки цитат: [SOURCE 1, 2, 3]
        list_citations = re.finditer(r"\[SOURCE ([\d,\s]+)\]", response_text)
        for m in list_citations:
            indices_str = m.group(1).split(",")
            for index in indices_str:
                try:
                    cited_indices.add(int(index.strip()))
                except ValueError:
                    pass  # Игнорируем некорректные значения

        logger.debug("[DEBUG: Citations] Found indices: %s", cited_indices)

        for i, chunk in enumerate(chunks):
            chunk["source_id"] = i + 1
            chunk["cited"] = (i + 1) in cited_indices
            chunk["excerpt"] = chunk["text"][:200] + "..."
        return chunks

    def _calculate_confidence(self, sources: List[Dict], response_text: str) -> float:
        if not sources:
            logger.debug("[DEBUG: Confidence] No sources provided, returning 0.0")
            return 0.0

        cited_sources = [s for s in sources if s.get("cited")]
        if not cited_sources:
            logger.debug("[DEBUG: Confidence] No cited sources, returning 0.1")
            return 0.1

        # --- Rerank Score Calculation ---
        rerank_scores = [s.get("rerank_score", 0.0) for s in cited_sources]
        avg_rerank_score = sum(rerank_scores) / len(rerank_scores)

        is_fallback_score = False
        if avg_rerank_score <= 0.0:
            similarities = [s.get("similarity", 0.0) for s in cited_sources]
            avg_rerank_score = (
                sum(similarities) / len(similarities) if similarities else 0.0
            )
            is_fallback_score = True
            logger.debug(
                "[DEBUG: Confidence] Used fallback similarity score: %.4f",
                avg_rerank_score,
            )

        # --- Citation Ratio ---
        citation_ratio = len(cited_sources) / len(sources)

        # --- Penalty for uncertainty ---
        penalty = 0.0
        lower_response = response_text.lower()
        if any(
            phrase in lower_response
            for phrase in ["not found", "не найдено", "недостаточно информации"]
        ):
            penalty = 0.4

        # --- Base Confidence ---
        base_confidence = 0.3

        # --- Final Calculation ---
        confidence_formula = (avg_rerank_score * 0.6 + citation_ratio * 0.4) - penalty
        confidence = max(base_confidence, confidence_formula)
        final_confidence = cast(float, max(0.0, min(1.0, confidence)))

        logger.debug(
            "--- [DEBUG: Confidence Calculation] ---\n"
            "Cited sources: %d out of %d\n"
            "Average Rerank Score: %.4f (Fallback used: %s)\n"
            "Citation Ratio: %.4f\n"
            "Uncertainty Penalty: %.2f\n"
            "Base Confidence: %.2f\n"
            "Pre-clamp Confidence (Formula): %.4f\n"
            "Final Confidence: %.4f\n"
            "--- [END DEBUG] ---",
            len(cited_sources),
            len(sources),
            avg_rerank_score,
            is_fallback_score,
            citation_ratio,
            penalty,
            base_confidence,
            confidence_formula,
            final_confidence,
        )

        return final_confidence

    def _generate_fallback_response(
        self,
        chunks: List[Dict],
        warning: str,
        embed_time_ms: float,
        confidence: float = 0.0,
        total_time: float = 0.0,
    ) -> Dict:
        # В fallback-режиме остальные тайминги нерелевантны или равны нулю
        timings = {
            "embed": embed_time_ms,
            "retrieve": 0,
            "rerank": 0,
            "llm": 0,
            "total": total_time * 1000,
        }

        return {
            "response_md": "К сожалению, я не могу дать уверенный ответ на основе доступной информации. Пожалуйста, проверьте следующие наиболее релевантные источники.",
            "confidence_score": confidence,
            "sources": chunks,
            "timings_ms": timings,
            "warnings": [warning, "fallback"],
        }

    def _log_chunks(self, chunks: List[Dict], stage: str, score_key: str):
        logger.debug("--- [DEBUG: Chunks at '%s' stage] ---", stage)
        if not chunks:
            logger.debug("No chunks to display.")
            return

        for i, chunk in enumerate(chunks[:10]):  # Логируем только топ-10 для краткости
            score = chunk.get(score_key, 0.0)
            logger.debug(
                "  - Chunk %2d (score: %.4f): %s...",
                i + 1,
                score,
                chunk["text"][:100],
            )
        if len(chunks) > 10:
            logger.debug("  ... and %d more chunks.", len(chunks) - 10)
        logger.debug("--- [END DEBUG] ---")
