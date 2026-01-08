import re
from typing import Any, Dict, List

from transformers import AutoTokenizer


class MarkdownChunker:
    """
    "Умный" чанкер для Markdown-документов.
    Разбивает текст по заголовкам и сохраняет целостность блоков кода.
    """
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        model_name: str = "BAAI/bge-m3"
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _count_tokens(self, text: str) -> int:
        """Подсчитывает количество токенов в тексте."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk(self, document_text: str, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Основной метод, который разбивает документ на чанки.
        """
        chunks = []
        
        # 1. Разделение на секции по заголовкам ## и ###
        sections = re.split(r'(^#{2,3}\s.*$)', document_text, flags=re.MULTILINE)
        
        current_header = ""
        current_text = ""
        
        processed_sections = []
        for part in sections:
            if re.match(r'^#{2,3}\s.*$', part):
                if current_text.strip():
                    processed_sections.append((current_header, current_text.strip()))
                current_header = part.strip()
                current_text = ""
            else:
                current_text += part
        if current_text.strip():
            processed_sections.append((current_header, current_text.strip()))

        chunk_index = 0
        for header, text in processed_sections:
            if self._count_tokens(text) <= self.chunk_size:
                chunks.append({
                    "text": f"{header}\n{text}" if header else text,
                    "metadata": {"header": header, "chunk_index": chunk_index, **doc_metadata}
                })
                chunk_index += 1
                continue

            sub_chunks = self._split_text_with_code_awareness(text)
            
            for sub_chunk in sub_chunks:
                chunks.append({
                    "text": f"{header}\n{sub_chunk}" if header else sub_chunk,
                    "metadata": {"header": header, "chunk_index": chunk_index, **doc_metadata}
                })
                chunk_index += 1
        
        return self._apply_overlap(chunks)

    def _split_text_with_code_awareness(self, text: str) -> List[str]:
        parts = re.split(r'(```[\s\S]*?```)', text)
        chunks = []
        current_chunk = ""

        for part in parts:
            if part.startswith('```'):
                if self._count_tokens(part) > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    chunks.append(part)
                    current_chunk = ""
                else:
                    current_chunk += part
            else:
                sentences = re.split(r'(?<=[.!?])\s+', part)
                for sentence in sentences:
                    if self._count_tokens(current_chunk + sentence) > self.chunk_size:
                        chunks.append(current_chunk)
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return [c.strip() for c in chunks if c.strip()]

    def _apply_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.chunk_overlap or len(chunks) < 2:
            return [
                {**chunk, "token_count": self._count_tokens(chunk["text"])}
                for chunk in chunks
            ]

        overlapped_chunks = []
        for i in range(len(chunks)):
            current_chunk_data = chunks[i]
            final_text = current_chunk_data["text"]

            # Добавляем overlap слева
            if i > 0:
                prev_chunk_text = chunks[i-1]["text"]
                overlap_tokens = self.tokenizer.encode(prev_chunk_text, add_special_tokens=False)[-self.chunk_overlap:]
                overlap_text = self.tokenizer.decode(overlap_tokens, skip_special_tokens=True)
                final_text = f"{overlap_text}\n...\n{final_text}"

            # Добавляем overlap справа, контролируя размер
            if i < len(chunks) - 1:
                next_chunk_text = chunks[i+1]["text"]
                max_len = self.chunk_size + self.chunk_overlap
                
                overlap_suffix = "\n...\n" + self.tokenizer.decode(
                    self.tokenizer.encode(next_chunk_text, add_special_tokens=False)[:self.chunk_overlap]
                )
                
                # Добавляем суффикс и обрезаем, если нужно
                temp_text = final_text + overlap_suffix
                
                # Простой строковый срез, если превысили лимит.
                # Это не идеально, но сохраняет маркеры.
                while self._count_tokens(temp_text) > max_len:
                    temp_text = temp_text[:-10] # Обрезаем по 10 символов
                
                final_text = temp_text

            final_chunk_data = {
                "text": final_text,
                "metadata": current_chunk_data["metadata"],
                "token_count": self._count_tokens(final_text)
            }
            overlapped_chunks.append(final_chunk_data)

        return overlapped_chunks