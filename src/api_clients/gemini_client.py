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

def _extract_gemini_text(resp: Any) -> str:
    # 1) Fast path
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    # 2) Candidates/parts path
    candidates = getattr(resp, "candidates", None) or []
    for c in candidates:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) or []
        buf = []
        for p in parts:
            t = getattr(p, "text", None)
            if isinstance(t, str):
                buf.append(t)
        s = "".join(buf).strip()
        if s:
            return s

    # 3) Nothing usable
    return ""

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

            def _target_tokens(prefix: str, suffix: str, base_cap: int) -> int:
                # Approximate 4 chars/token, bias with +16 to reach boundary
                need = (min(len(suffix), 160) // 4) + 16
                return min(base_cap, max(32, need))

            # Try to extract prefix/suffix from the last user message content
            user_content = ""
            for m in reversed(messages):
                if m.role == "user":
                    user_content = m.content or ""
                    break
            max_tok = min(self.model_options.max_tokens, 128)
            try:
                if "<prefix/>" in user_content and "</prefix/>" in user_content and "<suffix/>" in user_content and "</suffix/>" in user_content:
                    pfx = user_content.split("<prefix/>\n", 1)[1].split("\n</prefix/>", 1)[0]
                    sfx = user_content.split("<suffix/>\n", 1)[1].split("\n</suffix/>", 1)[0]
                    max_tok = _target_tokens(pfx, sfx, min(self.model_options.max_tokens, 128))
                elif "<mask/>" in user_content:
                    # Fallback heuristic: split around mask for legacy template
                    parts = user_content.split("<mask/>")
                    pfx = parts[0]
                    sfx = parts[1] if len(parts) > 1 else ""
                    max_tok = _target_tokens(pfx, sfx, min(self.model_options.max_tokens, 128))
            except Exception:
                pass
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    temperature=min(self.model_options.temperature, 0.4),
                    top_p=self.model_options.top_p,
                    presence_penalty=self.model_options.presence_penalty,
                    frequency_penalty=self.model_options.frequency_penalty,
                    max_output_tokens=max_tok,
                    candidate_count=1,
                    # stop_sequences=["\n\n", "\n- ", "\n1. "],  # Uncomment if supported in your SDK
                    # response_mime_type="text/plain",          # Uncomment if supported
                ),
            )
            text = _extract_gemini_text(resp).strip()
            if not text:
                fb = getattr(resp, "prompt_feedback", None)
                br = getattr(fb, "block_reason", None)
                fr = None
                try:
                    c0 = (getattr(resp, "candidates", None) or [None])[0]
                    fr = getattr(c0, "finish_reason", None)
                except Exception:
                    pass
                detail = f" (block_reason={br}, finish_reason={fr})" if (br or fr) else ""
                return err(RuntimeError("Empty result from Gemini" + detail))
            return ok(text)
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


