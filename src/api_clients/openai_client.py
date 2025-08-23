from __future__ import annotations
from typing import List
from ..types import ApiClient, ChatMessage
from ..utils import make_api_request, Result, ok, err, get_http_client
import json
from ..settings import Settings


def _to_input_items(messages: List[ChatMessage]) -> list[dict]:
    items = []
    for m in messages:
        items.append({
            "role": m.role,
            "content": [{"type": "input_text", "text": m.content}],
        })
    return items


def _extract_output_text(payload: dict) -> str:
    # Prefer aggregated field if present
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    out = payload.get("output", []) or []
    texts: list[str] = []
    for item in out:
        if item.get("type") == "output_text" and "text" in item:
            texts.append(item["text"])
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if isinstance(c, dict) and "text" in c:
                    texts.append(c["text"])
    if not texts and "content" in payload:
        for c in payload["content"]:
            if isinstance(c, dict) and c.get("type") == "output_text":
                texts.append(c.get("text", ""))
    return "".join(texts)


class OpenAIClient(ApiClient):
    def __init__(self, api_key: str, url: str, model: str, model_options):
        self.api_key = api_key
        self.url = url
        self.model = model
        self.model_options = model_options

    @classmethod
    def from_settings(cls, s: Settings) -> "OpenAIClient":
        return cls(s.openai.key, s.openai.url, s.openai.model, s.model_options)

    async def query_chat_model(self, messages: List[ChatMessage]) -> Result[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "model": self.model,
            "input": _to_input_items(messages),
            "temperature": self.model_options.temperature,
            "top_p": self.model_options.top_p,
            "frequency_penalty": self.model_options.frequency_penalty,
            "presence_penalty": self.model_options.presence_penalty,
            "max_output_tokens": self.model_options.max_tokens,
        }
        data = await make_api_request(self.url, "POST", body, headers)
        if data.is_err(): return data
        try:
            content = _extract_output_text(data.value)
            return ok(content)
        except Exception as e:
            return err(e)

    async def stream_chat_model(self, messages: List[ChatMessage]):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }
        body = {
            "model": self.model,
            "input": _to_input_items(messages),
            "temperature": self.model_options.temperature,
            "top_p": self.model_options.top_p,
            "frequency_penalty": self.model_options.frequency_penalty,
            "presence_penalty": self.model_options.presence_penalty,
            "stream": True,
        }
        client = await get_http_client()
        async with client.stream("POST", self.url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                # terminate stream on error
                return
            async for raw_line in resp.aiter_lines():
                if not raw_line:
                    continue
                if raw_line.startswith("data: "):
                    data = raw_line[6:].strip()
                    if data in ("[DONE]", "done", "null"):
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    t = obj.get("type") or obj.get("event")
                    if t in ("response.output_text.delta", "output_text.delta"):
                        delta = obj.get("delta")
                        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
                            yield delta["text"]
                        elif isinstance(obj.get("output_text"), str):
                            yield obj["output_text"]
                    elif t == "message":
                        for c in obj.get("content", []) or []:
                            if isinstance(c, dict) and isinstance(c.get("text"), str):
                                yield c["text"]
                    elif t in ("response.completed", "error"):
                        break

    async def check_config(self) -> list[str]:
        errors: list[str] = []
        if not self.url: errors.append("OpenAI Responses API url is not set")
        if not self.api_key: errors.append("OpenAI API key is not set")
        if errors: return errors
        res = await self.query_chat_model([ChatMessage(role="user", content="Say hello world and nothing else.")])
        if res.is_err(): errors.append(str(res.error))
        return errors


