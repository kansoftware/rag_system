import asyncio

from typing import cast
import httpx

from src.config import settings

# Глобальный лок для обеспечения последовательного доступа к LLM
llm_lock = asyncio.Lock()

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
                "Authorization": f"Bearer {self.api_key}"
            },
            timeout=self.timeout
        )
        print(f"LLM Client initialized for model '{self.model}' at '{self.base_url}'")

    async def generate(self, prompt: str, temperature: float) -> str:
        """
        Отправляет запрос к LLM для генерации текста.
        """
        async with llm_lock:
            print("LLM lock acquired. Generating response...")
            try:
                response = await self.client.post(
                    "/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful technical assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": 2048,
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print("LLM response generated successfully.")
                return cast(str, content)

            except httpx.HTTPStatusError as e:
                print(f"Error communicating with LLM: {e.response.status_code} - {e.response.text}")
                return "Error: Could not get a response from the language model."
            except Exception as e:
                print(f"An unexpected error occurred in LLM client: {e}")
                return "Error: An unexpected error occurred while communicating with the language model."

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
        print("LLM client closed.")