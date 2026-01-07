import re
import time
from typing import Any, Dict, List, cast

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.models import Chunk, Document
from src.ingestion.embedding import EmbeddingModel

from .llm import LLMClient
from .reranker import RerankerModel


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

    async def query(
        self,
        db: Session,
        query_text: str,
        query_embedding: List[float],
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
            return self._generate_fallback_response([], "No relevant documents found.")

        reranked_chunks = self.reranker_model.rerank(query_text, candidates)
        final_chunks = reranked_chunks[:top_k_final]
        rerank_time = time.time() - start_time - retrieve_time

        prompt = self._build_prompt(query_text, final_chunks)

        llm_start_time = time.time()
        llm_response_text = await self.llm.generate(prompt, temperature)
        llm_time = time.time() - llm_start_time

        verified_sources = self._verify_citations(llm_response_text, final_chunks)
        confidence = self._calculate_confidence(verified_sources, llm_response_text)

        total_time = time.time() - start_time
        
        if confidence < min_confidence:
            return self._generate_fallback_response(
                final_chunks,
                f"Confidence score {confidence:.2f} is below threshold {min_confidence}.",
                confidence,
                total_time
            )

        return {
            "response_md": llm_response_text,
            "confidence_score": confidence,
            "sources": verified_sources,
            "timings_ms": {
                "retrieve": retrieve_time * 1000,
                "rerank": rerank_time * 1000,
                "llm": llm_time * 1000,
                "total": total_time * 1000,
            }
        }

    def _vector_search(self, db: Session, query_embedding: List[float], top_k: int) -> List[Dict]:
        results = db.query(
            Chunk,
            (Chunk.embedding.l2_distance(query_embedding)).label('distance')
        ).join(Document).order_by(text('distance asc')).limit(top_k).all()
        
        return [
            {
                "chunk_id": r.Chunk.id,
                "document_id": r.Chunk.document_id,
                "text": r.Chunk.chunk_text,
                "title": r.Chunk.document.title,
                "url": r.Chunk.document.source_url,
                "similarity": 1 - r.distance,
            } for r in results
        ]
        
    def _build_prompt(self, query: str, chunks: List[Dict]) -> str:
        context = "\n\n---\n\n".join([
            f"[SOURCE {i+1}] (Title: {c.get('title')}, URL: {c.get('url')})\n{c['text']}"
            for i, c in enumerate(chunks)
        ])
        
        return f"""You are a technical documentation assistant. Answer ONLY based on the provided sources.
STRICT RULES:
1. NEVER make up information.
2. ALWAYS cite sources using [SOURCE N] format for each statement.
3. If information is not in sources, say "Information not found in the provided sources."

USER QUESTION: {query}

PROVIDED SOURCES:
{context}

ANSWER (with citations):"""

    def _verify_citations(self, response_text: str, chunks: List[Dict]) -> List[Dict]:
        cited_indices = {int(m.group(1)) for m in re.finditer(r'\[SOURCE (\d+)\]', response_text)}
        
        for i, chunk in enumerate(chunks):
            chunk["source_id"] = i + 1
            chunk["cited"] = (i + 1) in cited_indices
            chunk["excerpt"] = chunk["text"][:200] + "..."
        return chunks

    def _calculate_confidence(self, sources: List[Dict], response_text: str) -> float:
        if not sources:
            return 0.0
            
        cited_sources = [s for s in sources if s.get("cited")]
        if not cited_sources:
            return 0.1

        avg_rerank_score = sum(s.get('rerank_score', 0) for s in cited_sources) / len(cited_sources)
        citation_ratio = len(cited_sources) / len(sources)
        
        penalty = 0.0
        if any(phrase in response_text.lower() for phrase in ["not found", "unclear", "insufficient"]):
            penalty = 0.3
            
        confidence = (avg_rerank_score * 0.7 + citation_ratio * 0.3) - penalty
        return cast(float, max(0.0, min(1.0, confidence)))

    def _generate_fallback_response(self, chunks: List[Dict], warning: str, confidence: float = 0.0, total_time: float = 0.0) -> Dict:
        return {
            "response_md": "К сожалению, я не могу дать уверенный ответ на основе доступной информации. Пожалуйста, проверьте следующие наиболее релевантные источники.",
            "confidence_score": confidence,
            "sources": chunks,
            "timings_ms": {"total": total_time * 1000},
            "warnings": [warning, "fallback"]
        }