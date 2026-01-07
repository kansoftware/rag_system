from typing import Dict, List

import torch
from FlagEmbedding import FlagReranker

from src.config import settings


class RerankerModel:
    """
    Класс-обертка для модели ре-ранжирования BAAI/bge-reranker-v2-m3.
    """
    def __init__(self):
        self.model_name = settings.RERANKER_MODEL
        self.device = settings.RERANKER_DEVICE

        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA is not available for reranker. Falling back to CPU.")
            self.device = 'cpu'

        print(f"Initializing reranker model {self.model_name} on device '{self.device}'...")
        # use_fp16=True ускоряет вычисления на GPU
        self.model = FlagReranker(self.model_name, use_fp16=True if self.device == 'cuda' else False)
        print("Reranker model loaded successfully.")

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Переранжирует список чанков на основе их релевантности к запросу.
        """
        if not chunks:
            return []

        # Reranker ожидает пары [запрос, текст_чанка]
        pairs = [(query, chunk['text']) for chunk in chunks]
        
        print(f"Reranking {len(chunks)} candidates...")
        scores = self.model.compute_score(pairs, normalize=True)

        if scores is not None:
            # Добавляем rerank_score к каждому чанку
            for chunk, score in zip(chunks, scores, strict=True):
                chunk['rerank_score'] = float(score)

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