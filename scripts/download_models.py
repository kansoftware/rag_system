from sentence_transformers import SentenceTransformer
from src.config import settings

# EMBEDDING_MODEL = 'BAAI/bge-m3'
# # Заменяем модель ре-ранкера на альтернативную из sentence-transformers
# RERANKER_MODEL = 'sentence-transformers/ms-marco-MiniLM-L-12-v2'

def main():
    """Скачивает и кэширует необходимые модели."""
    print(f"Downloading embedding model: {settings.EMBEDDING_MODEL}...")
    SentenceTransformer(settings.EMBEDDING_MODEL)
    print("Embedding model cached successfully.")

    print(f"Downloading reranker model: {settings.RERANKER_MODEL}...")
    # Используем SentenceTransformer для загрузки новой модели
    SentenceTransformer(settings.RERANKER_MODEL)
    print("Reranker model cached successfully.")

    print("\nAll models are cached.")

if __name__ == "__main__":
    main()