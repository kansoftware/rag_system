from typing import List
import torch
from sentence_transformers import SentenceTransformer
from src.config import settings

class EmbeddingModel:
    """
    Класс-обертка для модели эмбеддингов BAAI/bge-m3.
    """
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA is not available. Falling back to CPU.")
            self.device = 'cpu'
            
        print(f"Initializing embedding model {self.model_name} on device '{self.device}'...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print("Embedding model loaded successfully.")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка текстов батчами.
        """
        if not texts:
            return []
            
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        print("Embeddings generated.")
        return embeddings.tolist()

_embedding_model = None

def get_embedding_model() -> EmbeddingModel:
    """Возвращает синглтон-экземпляр модели эмбеддингов."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model