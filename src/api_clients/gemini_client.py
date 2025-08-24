from __future__ import annotations
from typing import List, Any, AsyncIterator
import os

from ..types import ApiClient, ChatMessage
from ..utils import Result, ok, err
from ..settings import Settings


import dotenv
# SDK: pip/uv add google-genai
from google import genai  # type: ignore
from google.genai import types as gtypes  # type: ignore

dotenv.load_dotenv()

def _to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat-style messages into a single text prompt for Gemini.

    Using a simple stitched prompt avoids SDK type/version differences.
    """
    system_texts: list[str] = []
    dialogue: list[str] = []
    for m in messages:
        if m.role == "system":
            system_texts.append(m.content)
        elif m.role == "user":
            dialogue.append(f"User: {m.content}")
        else:
            dialogue.append(f"Assistant: {m.content}")
    header = "\n".join(system_texts).strip()
    body = "\n".join(dialogue).strip()
    if header and body:
        return header + "\n\n" + body + "\nAssistant:"
    if body:
        return body + "\nAssistant:"
    return header


class GeminiClient(ApiClient):
    def __init__(self, key: str, model: str, model_options):
        api_key = key or os.getenv("GOOGLE_API_KEY", "")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.model_options = model_options

    @classmethod
    def from_settings(cls, s: Settings) -> "GeminiClient":
        return cls(s.gemini.key, s.gemini.model, s.model_options)

    async def query_chat_model(self, messages: List[ChatMessage]) -> Result[str]:
        try:
            prompt = _to_prompt(messages)
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    temperature=self.model_options.temperature,
                    top_p=self.model_options.top_p,
                    presence_penalty=self.model_options.presence_penalty,
                    frequency_penalty=self.model_options.frequency_penalty,
                    max_output_tokens=self.model_options.max_tokens,
                    candidate_count=1,
                ),
            )
            return ok(getattr(resp, "text", "") or "")
        except Exception as e:
            return err(e)

    async def stream_chat_model(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        try:
            prompt = _to_prompt(messages)
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    temperature=self.model_options.temperature,
                    top_p=self.model_options.top_p,
                    presence_penalty=self.model_options.presence_penalty,
                    frequency_penalty=self.model_options.frequency_penalty,
                    max_output_tokens=self.model_options.max_tokens,
                    candidate_count=1,
                ),
            )
            async for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception:
            # Fall back to one-shot on error
            result = await self.query_chat_model(messages)
            if result.is_ok():
                yield result.value
            return

    async def check_config(self) -> list[str]:
        errors: list[str] = []
        try:
            if not self.client:  # pragma: no cover
                errors.append("Gemini client not initialized")
        except Exception as e:
            errors.append(str(e))
        if errors:
            return errors
        res = await self.query_chat_model([ChatMessage(role="user", content="Say hello world and nothing else.")])
        return [] if res.is_ok() else [str(res.error)]


