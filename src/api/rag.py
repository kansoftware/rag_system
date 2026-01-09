import re
import asyncio
import time
from typing import Any, Dict, List, cast

from transformers import AutoTokenizer
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.models import Chunk, Document
from src.ingestion.embedding import EmbeddingModel

from .llm import LLMClient
from .reranker import RerankerModel
from src.config import settings


class RAGEngine:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        reranker_model: RerankerModel,
        llm_client: LLMClient
    ):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.llm = llm_client
        self.tokenizer = AutoTokenizer.from_pretrained(reranker_model.model_name) # Используем ту же модель, что и ре-ранкер/эмбеддер

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
            return self._generate_fallback_response([], "No relevant documents found.", embed_time_ms=embed_time_ms)

        self._log_chunks(candidates, "Initial retrieval", "similarity")
        
        rerank_time_start = time.time()

        if settings.ENABLE_RERANKER:
            reranked_chunks = await asyncio.to_thread(self.reranker_model.rerank, query_text, candidates)
            self._log_chunks(reranked_chunks, "After Reranking", "rerank_score")
        else:
            print("\n--- [INFO] Reranking is disabled. Using similarity scores. ---\n")
            # Для совместимости с остальным кодом, который ожидает 'rerank_score'
            for chunk in candidates:
                chunk['rerank_score'] = chunk.get('similarity', 0.0)
            reranked_chunks = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        # Сначала фильтруем по порогу, потом берем топ
        confident_chunks = [chunk for chunk in reranked_chunks if chunk.get('rerank_score', 0.0) > 0.7]
        final_chunks = confident_chunks[:top_k_final]
        rerank_time = time.time() - rerank_time_start

        prompt = self._build_prompt(query_text, final_chunks)

        llm_start_time = time.time()
        llm_response_text = await self.llm.generate(prompt, temperature)
        llm_time = time.time() - llm_start_time

        verified_sources = self._verify_citations(llm_response_text, final_chunks)
        confidence = self._calculate_confidence(verified_sources, llm_response_text)
        
        print(f"Confidence: {confidence:.2f} (min_confidence: {min_confidence})")

        total_time = time.time() - start_time
        
        if confidence < min_confidence:
            return self._generate_fallback_response(
                chunks=final_chunks,
                warning=f"Confidence score {confidence:.2f} is below threshold {min_confidence}.",
                embed_time_ms=embed_time_ms,
                confidence=confidence,
                total_time=total_time
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
            }
        }

    def _vector_search(self, db: Session, query_embedding: List[float], top_k: int) -> List[Dict]:
        # Используем косинусное расстояние, так как индекс создан с vector_cosine_ops
        results = db.query(
            Chunk,
            (Chunk.embedding.cosine_distance(query_embedding)).label('distance')
        ).join(Document).order_by(text('distance asc')).limit(top_k).all()
        
        return [
            {
                "chunk_id": r.Chunk.id,
                "document_id": r.Chunk.document_id,
                "text": r.Chunk.chunk_text,
                "title": r.Chunk.document.title,
                "url": r.Chunk.document.source_url,
                "similarity": max(0.0, 1.0 - float(r.distance)),
            } for r in results
        ]
        
    def _build_prompt(self, query: str, chunks: List[Dict]) -> str:
        context = "\n\n---\n\n".join([
            f"[SOURCE {i+1}] (Title: {c.get('title')}, URL: {c.get('url')})\n{c['text']}"
            for i, c in enumerate(chunks)
        ])

        prompt = f"""You are a world-class technical documentation assistant.
Your task is to answer user questions based ONLY on the provided information.

**// STRICT RULES //**
1.  **NEVER INVENT INFORMATION.** You must ground every single statement in the PROVIDED SOURCES.
2.  **CITE EACH SENTENCE.** Every sentence you write must end with a citation, like `[SOURCE 1]` or `[SOURCE 1, 2]`.
3.  **SYNTHESIZE, DON'T COPY.** Explain concepts in your own words. If the user asks for a code example, write a clear, working example based on the information from the sources.
4.  **HANDLE MISSING INFORMATION.** If and only if the sources do not contain any relevant information to answer the question, you must respond with the single sentence: "Information not found in the provided sources." Do not add any other text.

**// EXAMPLE OF A GOOD ANSWER //**
*USER QUESTION:* How do I use boost::bimap?

*YOUR ANSWER:*
Boost.Bimap provides a bidirectional map, allowing lookups by either key or value [SOURCE 2]. To use it, include the `<boost/bimap.hpp>` header [SOURCE 1].

Here is a basic example:
```cpp
#include <boost/bimap.hpp>
#include <iostream>
#include <string>

int main() {{
    boost::bimap<int, std::string> bm;
    bm.insert({{ 1, "one" }});
    bm.insert({{ 2, "two" }});

    // Find by key
    std::cout << bm.left.at(1) << std::endl; // "one"

    // Find by value
    std::cout << bm.right.at("two") << std::endl; // 2
}}
```
This example demonstrates creating a bimap and accessing its left and right views for lookups [SOURCE 1, 3]. More complex examples can be found in the Boost documentation [SOURCE 4].
**// END OF EXAMPLE //**

**// USER'S TASK //**
USER QUESTION: {query}

PROVIDED SOURCES:
---
{context}
---

ANSWER (with citations, following all rules and the example format):"""

        # --- Detailed Logging for Prompt ---
        print("\n--- [DEBUG: LLM Prompt] ---")
        print(f"User Query: {query}")
        print(f"Number of chunks provided: {len(chunks)}")
        # Больше не логируем контент, чтобы избежать путаницы
        for i, chunk in enumerate(chunks):
            token_count = len(self.tokenizer.encode(chunk['text']))
            print(f"  - Chunk {i+1} (rerank_score={chunk.get('rerank_score', 0.0):.4f}, tokens={token_count})")
        print(f"Total tokens in context: {len(self.tokenizer.encode(context))}")
        print("--- [END DEBUG] ---\n")
        
        return prompt

    def _verify_citations(self, response_text: str, chunks: List[Dict]) -> List[Dict]:
        cited_indices = set()
        
        # Ищем одиночные цитаты: [SOURCE 1]
        single_citations = re.finditer(r'\[SOURCE (\d+)\]', response_text)
        for m in single_citations:
            cited_indices.add(int(m.group(1)))
            
        # Ищем списки цитат: [SOURCE 1, 2, 3]
        list_citations = re.finditer(r'\[SOURCE ([\d,\s]+)\]', response_text)
        for m in list_citations:
            indices_str = m.group(1).split(',')
            for index in indices_str:
                try:
                    cited_indices.add(int(index.strip()))
                except ValueError:
                    pass # Игнорируем некорректные значения

        print(f"[DEBUG: Citations] Found indices: {cited_indices}")

        for i, chunk in enumerate(chunks):
            chunk["source_id"] = i + 1
            chunk["cited"] = (i + 1) in cited_indices
            chunk["excerpt"] = chunk["text"][:200] + "..."
        return chunks

    def _calculate_confidence(self, sources: List[Dict], response_text: str) -> float:
        if not sources:
            print("[DEBUG: Confidence] No sources provided, returning 0.0")
            return 0.0

        cited_sources = [s for s in sources if s.get("cited")]
        if not cited_sources:
            print("[DEBUG: Confidence] No cited sources, returning 0.1")
            return 0.1

        # --- Rerank Score Calculation ---
        rerank_scores = [s.get('rerank_score', 0.0) for s in cited_sources]
        avg_rerank_score = sum(rerank_scores) / len(rerank_scores)
        
        is_fallback_score = False
        if avg_rerank_score <= 0.0:
            similarities = [s.get('similarity', 0.0) for s in cited_sources]
            avg_rerank_score = sum(similarities) / len(similarities) if similarities else 0.0
            is_fallback_score = True
            print(f"[DEBUG: Confidence] Used fallback similarity score: {avg_rerank_score:.4f}")

        # --- Citation Ratio ---
        citation_ratio = len(cited_sources) / len(sources)

        # --- Penalty for uncertainty ---
        penalty = 0.0
        lower_response = response_text.lower()
        if any(phrase in lower_response for phrase in ["not found", "не найдено", "недостаточно информации"]):
            penalty = 0.4

        # --- Base Confidence ---
        base_confidence = 0.3

        # --- Final Calculation ---
        confidence_formula = (avg_rerank_score * 0.6 + citation_ratio * 0.4) - penalty
        confidence = max(base_confidence, confidence_formula)
        final_confidence = cast(float, max(0.0, min(1.0, confidence)))

        # --- Detailed Logging ---
        print("\n--- [DEBUG: Confidence Calculation] ---")
        print(f"Cited sources: {len(cited_sources)} out of {len(sources)}")
        print(f"Average Rerank Score: {avg_rerank_score:.4f} (Fallback used: {is_fallback_score})")
        print(f"Citation Ratio: {citation_ratio:.4f}")
        print(f"Uncertainty Penalty: {penalty:.2f}")
        print(f"Base Confidence: {base_confidence:.2f}")
        print(f"Pre-clamp Confidence (Formula): {confidence_formula:.4f}")
        print(f"Final Confidence: {final_confidence:.4f}")
        print("--- [END DEBUG] ---\n")

        return final_confidence

    def _generate_fallback_response(self, chunks: List[Dict], warning: str, embed_time_ms: float, confidence: float = 0.0, total_time: float = 0.0) -> Dict:
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
            "warnings": [warning, "fallback"]
        }

    def _log_chunks(self, chunks: List[Dict], stage: str, score_key: str):
        print(f"\n--- [DEBUG: Chunks at '{stage}' stage] ---")
        if not chunks:
            print("No chunks to display.")
            return
            
        for i, chunk in enumerate(chunks[:10]):  # Логируем только топ-10 для краткости
            score = chunk.get(score_key, 0.0)
            print(f"  - Chunk {i+1:2d} (score: {score:.4f}): {chunk['text'][:100]}...")
        if len(chunks) > 10:
            print(f"  ... and {len(chunks) - 10} more chunks.")
        print("--- [END DEBUG] ---\n")