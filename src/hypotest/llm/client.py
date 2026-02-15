# hypotest/llm/client.py

from openai import OpenAI
from typing import Optional
from .base import BaseLLMClient


class OpenAICompatibleClient(BaseLLMClient):
    """
    Works with any OpenAI-compatible provider:
    - OpenAI
    - Groq
    - TogetherAI
    - OpenRouter
    - Fireworks
    - vLLM
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        super().__init__(api_key, base_url, model)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,  # critical for compatibility
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content
