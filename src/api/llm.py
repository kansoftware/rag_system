import asyncio
import logging
from typing import cast

import httpx

from src.config import settings

# Глобальный семафор для ограничения параллельных запросов к LLM
llm_lock = asyncio.Semaphore(settings.LLM_MAX_CONCURRENT_REQUESTS)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Клиент для взаимодействия с OpenAI-совместимым API.
    Обеспечивает последовательную обработку запросов к LLM.
    """

    def __init__(self):
        self.base_url = settings.LLM_BASE_URL
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.timeout = settings.LLM_TIMEOUT

        if "openrouter" in self.base_url:
            self.provider = "openrouter"
        else:
            self.provider = "lmstudio"

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        logger.info(
            "LLM Client initialized for model '%s' at '%s' provider %s",
            self.model,
            self.base_url,
            self.provider,
        )

    async def generate(self, prompt: str, temperature: float) -> str:
        """
        Отправляет запрос к LLM для генерации текста.
        """
        async with llm_lock:
            logger.debug("LLM lock acquired. Generating response...")
            try:
                request_body = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful technical assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": 2048,
                }
                logger.debug(
                    "LLM request: model=%s, temperature=%.2f", self.model, temperature
                )

                response = await self.client.post(
                    "/chat/completions",
                    json=request_body,
                )
                response.raise_for_status()

                data = response.json()

                content = data["choices"][0]["message"]["content"]
                logger.debug("LLM response generated successfully.")
                return cast(str, content)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "Error communicating with LLM: %s - %s",
                    e.response.status_code,
                    e.response.text,
                )
                raise
            except Exception:
                logger.exception("An unexpected error occurred in LLMClient")
                raise

    async def close(self):
        await self.client.aclose()


_llm_client = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


async def close_llm_client():
    global _llm_client
    if _llm_client:
        await _llm_client.close()
        logger.info("LLM client closed.")
