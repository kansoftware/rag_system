from typing import List
from pathlib import Path
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.db.models import Document, Chunk
from src.db.session import SessionLocal
from .dedup import Deduplicator, compute_content_hash
from .chunking import MarkdownChunker
from .embedding import get_embedding_model

class IngestionPipeline:
    def __init__(self):
        self.chunker = MarkdownChunker()
        self.embedding_model = get_embedding_model()
        self.db: Session = SessionLocal()
        self.deduplicator = Deduplicator(self.db)

    def run(self, file_paths: List[Path], domain: str):
        """
        Запускает полный конвейер обработки и загрузки документов.
        """
        print(f"Starting ingestion for {len(file_paths)} files from domain '{domain}'...")
        
        new_docs_count = 0
        skipped_docs_count = 0
        total_chunks_count = 0

        for path in tqdm(file_paths, desc="Processing files"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                content_hash = compute_content_hash(content)

                if self.deduplicator.is_duplicate(content_hash):
                    skipped_docs_count += 1
                    continue

                document = Document(
                    file_path=str(path.resolve()),
                    source_url=None,
                    title=path.stem,
                    domain=domain,
                    content_hash=content_hash,
                    full_text=content,
                    metadata={"source": "markdown_files"}
                )
                self.db.add(document)
                self.db.flush() 

                doc_metadata = {"document_id": document.id, "domain": domain}
                chunks_data = self.chunker.chunk(content, doc_metadata)
                
                if not chunks_data:
                    continue

                chunk_texts = [c["text"] for c in chunks_data]
                embeddings = self.embedding_model.get_embeddings(chunk_texts)

                for i, chunk_data in enumerate(chunks_data):
                    chunk = Chunk(
                        document_id=document.id,
                        chunk_index=chunk_data["metadata"]["chunk_index"],
                        chunk_text=chunk_data["text"],
                        token_count=chunk_data["token_count"],
                        embedding=embeddings[i],
                        metadata=chunk_data["metadata"]
                    )
                    self.db.add(chunk)
                
                self.db.commit()
                self.deduplicator.add_hash(content_hash)
                new_docs_count += 1
                total_chunks_count += len(chunks_data)

            except Exception as e:
                print(f"Error processing file {path}: {e}")
                self.db.rollback()

        self.db.close()
        print("\n--- Ingestion Complete ---")
        print(f"New documents processed: {new_docs_count}")
        print(f"Duplicate documents skipped: {skipped_docs_count}")
        print(f"Total chunks created: {total_chunks_count}")