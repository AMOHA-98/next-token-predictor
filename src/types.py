from __future__ import annotations
from typing import List, Protocol, Literal, Callable, Dict, Any, AsyncIterator
from pydantic import BaseModel

Role = Literal["user", "assistant", "system"]


class ChatMessage(BaseModel):
    content: str
    role: Role


class ModelOptions(BaseModel):
    temperature: float = 0.2
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 256


class FewShotExample(BaseModel):
    context: str
    input: str
    answer: str


class ApiClient(Protocol):
    async def query_chat_model(self, messages: List[ChatMessage]) -> "Result[str]": ...
    async def check_config(self) -> List[str]: ...
    async def stream_chat_model(self, messages: List[ChatMessage]) -> AsyncIterator[str]: ...


class PredictionService(Protocol):
    async def fetch_predictions(self, prefix: str, suffix: str) -> "Result[str]": ...


UserMessageFormatter = Callable[[Dict[str, str]], str]


