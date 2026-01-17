from typing import Dict, List
import torch
from sentence_transformers import CrossEncoder
from pathlib import Path
import onnxruntime

from src.config import settings


class RerankerModel:
    """
    Класс-обертка для модели ре-ранжирования, использующая sentence-transformers.
    Поддерживает как стандартные модели PyTorch, так и оптимизированные ONNX-модели.
    """
    def __init__(self):
        from numpy import __version__ as numpy_version
        from packaging.version import parse

        self.model_name = settings.RERANKER_MODEL
        # Принудительно используем 'bge-reranker-base' для старого окружения,
        # так как 'bge-reranker-v2-m3' не имеет стандартных файлов pytorch_model.bin
        if parse(numpy_version) < parse("2.0.0"):
            print("Legacy environment detected (numpy < 2.0). Forcing reranker model to 'cross-encoder/ms-marco-MiniLM-L12-v2'.")
            # self.model_name = "BAAI/bge-reranker-base"
            self.model_name = "cross-encoder/ms-marco-MiniLM-L12-v2"

        self.device = settings.RERANKER_DEVICE
        self.batch_size = settings.RERANKER_BATCH_SIZE
        self.use_onnx = settings.RERANKER_ONNX

        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA is not available for reranker. Falling back to CPU.")
            self.device = 'cpu'

        model_to_load = self.model_name
        
        if self.use_onnx:
            print("ONNX runtime enabled for reranker.")
            try:
                from huggingface_hub import hf_hub_download
                # Используем hf_hub_download для поиска ONNX-модели в кеше
                onnx_path = hf_hub_download(repo_id=self.model_name, filename="model.onnx")
                model_to_load = onnx_path
                print(f"Loading ONNX reranker model from {model_to_load}...")

                # Проверяем, доступен ли CUDA провайдер для ONNX
                if self.device == 'cuda' and 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
                    print("Warning: ONNX runtime CUDA provider is not available. Falling back to CPU.")
                    self.device = 'cpu'
            except Exception as e:
                print(f"Warning: Could not download or find ONNX model. Falling back to PyTorch. Error: {e}")
                self.use_onnx = False

        print(f"Initializing reranker model {self.model_name} on device '{self.device}'...")
        # Если используем ONNX, device должен быть None, так как провайдер указывается при создании сессии
        device_for_encoder = self.device if not self.use_onnx else None
        self.model = CrossEncoder(model_to_load, device=device_for_encoder, automodel_args={'trust_remote_code': True})
        
        if self.use_onnx and hasattr(self.model, 'model') and hasattr(self.model.model, 'session'):
             # Явное указание провайдера для ONNX
            provider = 'CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider'
            self.model.model.session.set_providers([provider])

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
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=self.batch_size
        )

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