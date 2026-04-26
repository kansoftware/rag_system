import logging
from typing import List, cast

import torch
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Класс-обертка для модели эмбеддингов BAAI/bge-m3.
    """

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.batch_size = settings.EMBEDDING_BATCH_SIZE

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available. Falling back to CPU.")
            self.device = "cpu"

        logger.info(
            "Initializing embedding model %s on device '%s'...",
            self.model_name,
            self.device,
        )
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Embedding model loaded successfully.")

        # Определяем фактическую размерность эмбеддингов модели
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        # Валидация соответствия с конфигурацией
        if self.embedding_dim != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: model '{self.model_name}' produces "
                f"{self.embedding_dim}-dimensional vectors, but config EMBEDDING_DIM is {settings.EMBEDDING_DIM}. "
                "Please update EMBEDDING_DIM in .env to match the model's dimension."
            )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка текстов батчами.
        """
        if not texts:
            return []

        logger.info("Generating embeddings for %d chunks...", len(texts))

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        logger.info("Embeddings generated.")
        return cast(List[List[float]], embeddings.tolist())


_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Возвращает синглтон-экземпляр модели эмбеддингов."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
