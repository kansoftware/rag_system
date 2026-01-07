import re
from typing import Optional

import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Элементы и классы/ID для удаления
BLOCKLIST_TAGS = ['nav', 'footer', 'aside', 'script', 'style', 'header', 'form']
BLOCKLIST_PATTERNS = ['menu', 'sidebar', 'ad', 'banner', 'cookie', 'popup', 'promo', 'related', 'share', 'social']

class HTMLConverter:
    def __init__(self, keep_tables: bool = True, keep_images: bool = False):
        self.keep_tables = keep_tables
        self.keep_images = keep_images

    def convert(self, html_content: str, base_url: Optional[str] = None) -> str:
        """
        Полный цикл очистки HTML и конвертации в Markdown.
        """
        # 1. Попытка извлечь основной контент с помощью trafilatura
        main_content_html = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=self.keep_tables,
            include_images=self.keep_images,
            no_fallback=True,
            url=base_url
        )

        # 2. Если trafilatura не справилась, используем ручную очистку
        if not main_content_html:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Удаляем блочные элементы
            for tag in soup.find_all(BLOCKLIST_TAGS):
                tag.decompose()
            
            # Удаляем элементы по паттернам в class/id
            for pattern in BLOCKLIST_PATTERNS:
                # Используем re.compile для поиска по подстроке в атрибутах
                for elem in soup.find_all(class_=re.compile(pattern, re.IGNORECASE)):
                    elem.decompose()
                for elem in soup.find_all(id=re.compile(pattern, re.IGNORECASE)):
                    elem.decompose()
            
            # Ищем основной контейнер контента
            body = soup.find('article') or soup.find('main') or soup.find('body')
            main_content_html = str(body) if body else ''

        if not main_content_html:
            return ""

        # 3. Конвертация очищенного HTML в Markdown
        markdown_text = md(
            main_content_html, 
            heading_style="ATX",
            code_language_callback=self._get_code_language
        )
        
        return self._postprocess_markdown(markdown_text)

    def _get_code_language(self, el: BeautifulSoup) -> str:
        """Попытка определить язык программирования из классов элемента `<code>`."""
        lang_class = el.get('class')
        if lang_class and isinstance(lang_class, list):
            for cls in lang_class:
                if isinstance(cls, str) and cls.startswith('language-'):
                    return cls.replace('language-', '')
        return ''

    def _postprocess_markdown(self, text: str) -> str:
        """Дополнительная очистка Markdown после конвертации."""
        # Удаление лишних пустых строк (более 2 подряд)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Удаление пробелов в начале и конце строк
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()