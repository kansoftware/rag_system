from typing import Dict, List
import torch
from sentence_transformers import CrossEncoder

from src.config import settings


class RerankerModel:
    """
    Класс-обертка для модели ре-ранжирования, использующая sentence-transformers.
    Модель `ms-marco-MiniLM-L-12-v2` является cross-encoder'ом, который хорошо
    подходит для задач ре-ранжирования.
    """
    def __init__(self):
        self.model_name = settings.RERANKER_MODEL
        self.device = settings.RERANKER_DEVICE

        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA is not available for reranker. Falling back to CPU.")
            self.device = 'cpu'

        print(f"Initializing reranker model {self.model_name} on device '{self.device}'...")
        # Используем CrossEncoder для моделей ре-ранжирования
        self.model = CrossEncoder(self.model_name, device=self.device)
        print("Reranker model loaded successfully.")

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Переранжирует список чанков на основе их релевантности к запросу.
        """
        if not chunks:
            return []

        # sentence-transformers ожидает пары [запрос, текст_чанка]
        pairs = [(query, chunk['text']) for chunk in chunks if chunk.get('text')]
        
        if not pairs:
            print("Warning: No valid (query, text) pairs found to rerank.")
            for chunk in chunks:
                chunk['rerank_score'] = 0.0
            return chunks

        print(f"Reranking {len(pairs)} candidates with cross-encoder...")
        
        # Вычисляем оценки релевантности. Cross-encoder напрямую возвращает оценки.
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Сопоставляем оценки с чанками, которые были отправлены на обработку
        valid_chunks = [chunk for chunk in chunks if chunk.get('text')]
        for chunk, score in zip(valid_chunks, scores):
            chunk['rerank_score'] = float(score)

        # Для отфильтрованных (невалидных) чанков устанавливаем score в 0
        for chunk in chunks:
            if 'rerank_score' not in chunk:
                chunk['rerank_score'] = 0.0

        # Сортируем чанки по убыванию rerank_score
        reranked_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        
        print("Reranking complete.")
        return reranked_chunks

_reranker_model = None

def get_reranker_model() -> RerankerModel:
    """Возвращает синглтон-экземпляр модели ре-ранжирования."""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = RerankerModel()
    return _reranker_model