import hashlib
import unicodedata
import re
from sqlalchemy.orm import Session
from src.db.models import Document

def normalize_text(text: str) -> str:
    """
    Приводит текст к каноническому виду для вычисления хэша.
    """
    # NFKC-нормализация для обработки совместимых символов (например, 'ﬁ' -> 'fi')
    text = unicodedata.normalize('NFKC', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Замена всех пробельных символов на один пробел
    text = re.sub(r'\s+', ' ', text)
    # Удаление пробелов в начале и конце
    return text.strip()

def compute_content_hash(text: str) -> bytes:
    """
    Вычисляет 256-битный хэш blake2b от нормализованного текста.
    """
    normalized_text = normalize_text(text)
    # digest_size=32 для 256-битного хэша (32 байта)
    return hashlib.blake2b(normalized_text.encode('utf-8'), digest_size=32).digest()

class Deduplicator:
    """
    Отвечает за проверку дубликатов документов в базе данных.
    """
    def __init__(self, db_session: Session):
        self._db = db_session
        self._seen_hashes = self._load_existing_hashes()

    def _load_existing_hashes(self) -> set[bytes]:
        """Загружает все существующие хэши из БД для быстрой проверки в памяти."""
        print("Loading existing content hashes from the database...")
        hashes = self._db.query(Document.content_hash).all()
        # .all() возвращает список кортежей, извлекаем первый элемент
        return {h[0] for h in hashes}

    def is_duplicate(self, content_hash: bytes) -> bool:
        """
        Проверяет, был ли уже обработан документ с таким хэшом.
        """
        return content_hash in self._seen_hashes

    def add_hash(self, content_hash: bytes):
        """Добавляет новый хэш в кэш, чтобы избежать повторной проверки."""
        self._seen_hashes.add(content_hash)