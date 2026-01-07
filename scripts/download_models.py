from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

EMBEDDING_MODEL = 'BAAI/bge-m3'
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'

def main():
    """Скачивает и кэширует необходимые модели."""
    print(f"Downloading embedding model: {EMBEDDING_MODEL}...")
    SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model cached successfully.")

    print(f"Downloading reranker model: {RERANKER_MODEL}...")
    FlagReranker(RERANKER_MODEL)
    print("Reranker model cached successfully.")

    print("\nAll models are cached.")

if __name__ == "__main__":
    main()