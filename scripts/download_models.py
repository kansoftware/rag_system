from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = 'BAAI/bge-m3'
# Заменяем модель ре-ранкера на альтернативную из sentence-transformers
RERANKER_MODEL = 'sentence-transformers/ms-marco-MiniLM-L-12-v2'

def main():
    """Скачивает и кэширует необходимые модели."""
    print(f"Downloading embedding model: {EMBEDDING_MODEL}...")
    SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model cached successfully.")

    print(f"Downloading reranker model: {RERANKER_MODEL}...")
    # Используем SentenceTransformer для загрузки новой модели
    SentenceTransformer(RERANKER_MODEL)
    print("Reranker model cached successfully.")

    print("\nAll models are cached.")

if __name__ == "__main__":
    main()