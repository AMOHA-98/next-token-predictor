from __future__ import annotations
from typing import List
from ..types import ApiClient, ChatMessage
from ..utils import make_api_request, Result, ok, err, get_http_client
import json
from ..settings import Settings


class OpenRouterClient(ApiClient):
    def __init__(self, key: str, url: str, model: str, model_options, site_url: str | None, app_title: str | None):
        self.key = key
        self.url = url
        self.model = model
        self.model_options = model_options
        self.site_url = site_url
        self.app_title = app_title

    @classmethod
    def from_settings(cls, s: Settings) -> "OpenRouterClient":
        return cls(
            s.openrouter.key, s.openrouter.url, s.openrouter.model,
            s.model_options, getattr(s.openrouter, "site_url", None), getattr(s.openrouter, "app_title", None)
        )

    async def query_chat_model(self, messages: List[ChatMessage]) -> Result[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_title:
            headers["X-Title"] = self.app_title

        body = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "temperature": self.model_options.temperature,
            "top_p": self.model_options.top_p,
            "frequency_penalty": self.model_options.frequency_penalty,
            "presence_penalty": self.model_options.presence_penalty,
            "max_tokens": self.model_options.max_tokens,
        }
        data = await make_api_request(self.url, "POST", body, headers)
        if data.is_err():
            return data
        try:
            return ok(data.value["choices"][0]["message"]["content"])
        except Exception as e:
            return err(e)

    async def stream_chat_model(self, messages: List[ChatMessage]):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
            "Accept": "text/event-stream",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_title:
            headers["X-Title"] = self.app_title

        body = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            "temperature": self.model_options.temperature,
            "top_p": self.model_options.top_p,
            "frequency_penalty": self.model_options.frequency_penalty,
            "presence_penalty": self.model_options.presence_penalty,
        }
        client = await get_http_client()
        async with client.stream("POST", self.url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                return
            async for raw_line in resp.aiter_lines():
                if not raw_line:
                    continue
                if raw_line.startswith("data: "):
                    data = raw_line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    try:
                        delta = obj["choices"][0]["delta"].get("content", "")
                    except Exception:
                        delta = ""
                    if delta:
                        yield delta

    async def check_config(self) -> list[str]:
        errors: list[str] = []
        if not self.url:
            errors.append("OpenRouter API url is not set")
        if not self.key:
            errors.append("OpenRouter API key is not set")
        if errors:
            return errors
        res = await self.query_chat_model([ChatMessage(role="user", content="Say hello world and nothing else.")])
        if res.is_err():
            errors.append(str(res.error))
        return errors


