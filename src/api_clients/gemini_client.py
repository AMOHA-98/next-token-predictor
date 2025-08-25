from __future__ import annotations
from typing import List, Any, AsyncIterator, Tuple
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

def _to_contents(messages: List[ChatMessage]) -> Tuple[str, list[Any]]:
    """Convert chat-style messages into Gemini contents and system instruction."""
    system_texts: list[str] = []
    contents: list[Any] = []
    for m in messages:
        if m.role == "system":
            if m.content:
                system_texts.append(m.content)
        elif m.role == "user":
            contents.append(gtypes.Content(role="user", parts=[gtypes.Part(text=m.content or "")]))
        else:  # assistant
            contents.append(gtypes.Content(role="model", parts=[gtypes.Part(text=m.content or "")]))
    system_instruction = "\n\n".join(system_texts).strip()
    return system_instruction, contents


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
            system_instruction, contents = _to_contents(messages)

            def _target_tokens(prefix: str, suffix: str, base_cap: int) -> int:
                # Approximate 4 chars/token, bias with +24 to reach boundary
                need = (min(len(suffix), 200) // 4) + 24
                floor = 64 if len(suffix) == 0 else 48
                return min(base_cap, max(floor, need))

            # Try to extract prefix/suffix from the last user message content
            user_content = ""
            for m in reversed(messages):
                if m.role == "user":
                    user_content = m.content or ""
                    break
            max_tok = min(self.model_options.max_tokens, 128)
            sfx_for_stops = ""
            try:
                if "<prefix/>" in user_content and "</prefix/>" in user_content and "<suffix/>" in user_content and "</suffix/>" in user_content:
                    pfx = user_content.split("<prefix/>\n", 1)[1].split("\n</prefix/>", 1)[0]
                    sfx = user_content.split("<suffix/>\n", 1)[1].split("\n</suffix/>", 1)[0]
                    max_tok = _target_tokens(pfx, sfx, min(self.model_options.max_tokens, 128))
                    sfx_for_stops = sfx
                elif "<mask/>" in user_content:
                    # Fallback heuristic: split around mask for legacy template
                    parts = user_content.split("<mask/>")
                    pfx = parts[0]
                    sfx = parts[1] if len(parts) > 1 else ""
                    max_tok = _target_tokens(pfx, sfx, min(self.model_options.max_tokens, 128))
                    sfx_for_stops = sfx
            except Exception:
                pass

            # Build stop sequences: suffix head + generic boundaries
            stop: list[str] = []
            try:
                head16 = (sfx_for_stops or "")[:16].strip()
                head8  = (sfx_for_stops or "")[:8].strip()
                for h in (head16, head8):
                    if len(h) >= 2:
                        stop.append(h)
            except Exception:
                pass
            # Only add generic stops when there is a suffix
            if (sfx_for_stops or "").strip():
                stop.extend(["\n\n", "\n- ", "\n1. "])

            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents if contents else (messages[-1].content if messages else ""),
                config=gtypes.GenerateContentConfig(
                    temperature=min(self.model_options.temperature, 0.4),
                    top_p=self.model_options.top_p,
                    presence_penalty=self.model_options.presence_penalty,
                    frequency_penalty=self.model_options.frequency_penalty,
                    max_output_tokens=max_tok,
                    candidate_count=1,
                    system_instruction=system_instruction or None,
                    stop_sequences=stop,
                ),
            )
            text = _extract_gemini_text(resp)
            if text is None or text.strip() == "":
                return ok("")
            return ok(text.strip())
        except Exception as e:
            return err(e)

    async def stream_chat_model(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        try:
            system_instruction, contents = _to_contents(messages)
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents if contents else (messages[-1].content if messages else ""),
                config=gtypes.GenerateContentConfig(
                    temperature=self.model_options.temperature,
                    top_p=self.model_options.top_p,
                    presence_penalty=self.model_options.presence_penalty,
                    frequency_penalty=self.model_options.frequency_penalty,
                    max_output_tokens=self.model_options.max_tokens,
                    candidate_count=1,
                    system_instruction=system_instruction or None,
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


