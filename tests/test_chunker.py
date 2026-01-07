import pytest

from src.ingestion.chunking import MarkdownChunker


@pytest.fixture(scope="module")
def chunker():
    # Используем модель, которая не требует скачивания, для ускорения тестов
    return MarkdownChunker(chunk_size=100, chunk_overlap=20, model_name="distilbert-base-uncased")

def test_simple_chunking(chunker):
    text = ("Это первое предложение. " * 20) + ("Это второе предложение. " * 20) + ("Это третье. " * 20)
    chunks = chunker.chunk(text, {})
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["token_count"] <= chunker.chunk_size + chunker.chunk_overlap

def test_code_block_is_not_split(chunker):
    text = """
Это текст до блока кода.
```python
def hello_world():
    # Этот блок кода достаточно длинный,
    # чтобы потенциально быть разделенным,
    # но он должен остаться целым.
    print("Hello, World!")
    a = list(range(100))
```
Это текст после блока кода.
"""
    chunks = chunker.chunk(text, {})
    # Ожидаем, что блок кода будет в одном чанке
    code_chunk_found = False
    for chunk in chunks:
        if "def hello_world():" in chunk["text"]:
            assert "```python" in chunk["text"]
            assert "```" in chunk["text"]
            code_chunk_found = True
            break
    assert code_chunk_found, "Блок кода не был найден в одном чанке"

def test_header_separation(chunker):
    text = "## Секция 1\nТекст первой секции, достаточно длинный, чтобы быть уверенным, что он не сольется со второй. " * 10 + "\n\n## Секция 2\nТекст второй секции."
    chunks = chunker.chunk(text, {})
    
    # Проверяем, что чанки начинаются с правильных заголовков
    assert "## Секция 1" in chunks[0]['text']
    # В зависимости от длины, второй заголовок может быть во втором или последующих чанках
    assert any("## Секция 2" in chunk['text'] for chunk in chunks)
    # Убедимся, что один чанк не содержит оба заголовка
    for chunk in chunks:
        assert not ("## Секция 1" in chunk['text'] and "## Секция 2" in chunk['text'])

def test_no_overlap_for_single_chunk(chunker):
    text = "Короткий текст, который помещается в один чанк."
    chunks = chunker.chunk(text, {})
    assert len(chunks) == 1
    assert "..." not in chunks[0]['text'] # Проверяем отсутствие маркеров overlap

def test_overlap_application(chunker):
    long_text_part1 = "Это первая часть очень длинного текста. " * 30
    long_text_part2 = "Это вторая часть очень длинного текста. " * 30
    text = long_text_part1 + long_text_part2
    
    chunks = chunker.chunk(text, {})
    assert len(chunks) > 1
    
    # Проверяем, что у второго чанка есть overlap с первым
    if len(chunks) > 1:
        # Просто проверяем, что маркер наложения присутствует,
        # так как точное содержимое может варьироваться.
        assert "..." in chunks[1]['text']
