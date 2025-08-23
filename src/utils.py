from __future__ import annotations
import random
from typing import Any, Dict, Generic, TypeVar, Callable
import httpx

T = TypeVar("T")


class Result(Generic[T]):
    __slots__ = ("_ok", "_value", "_error")

    def __init__(self, ok: bool, value: T | None = None, error: Exception | None = None):
        self._ok, self._value, self._error = ok, value, error

    def is_ok(self) -> bool:
        return self._ok

    def is_err(self) -> bool:
        return not self._ok

    @property
    def value(self) -> T:
        if not self._ok:
            raise self._error  # type: ignore[misc]
        return self._value  # type: ignore[return-value]

    @property
    def error(self) -> Exception:
        if self._ok:
            raise RuntimeError("No error")
        return self._error  # type: ignore[return-value]

    def map(self, fn: Callable[[T], T]) -> "Result[T]":
        return ok(fn(self._value)) if self._ok else self  # type: ignore[arg-type]


def ok(value: T) -> Result[T]:
    return Result(True, value=value)


def err(error: Exception) -> Result[Any]:
    return Result(False, error=error)


def generate_random_string(n: int) -> str:
    chars = "0123456789abcdef"
    return "".join(random.choice(chars) for _ in range(n))


_HTTP_CLIENT: httpx.AsyncClient | None = None


async def init_http_client(timeout: float = 30.0) -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        transport = httpx.AsyncHTTPTransport(retries=2)
        _HTTP_CLIENT = httpx.AsyncClient(timeout=timeout, transport=transport)


async def get_http_client() -> httpx.AsyncClient:
    if _HTTP_CLIENT is None:
        await init_http_client()
    assert _HTTP_CLIENT is not None
    return _HTTP_CLIENT


async def close_http_client() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        try:
            await _HTTP_CLIENT.aclose()
        finally:
            _HTTP_CLIENT = None


async def make_api_request(
    url: str,
    method: str,
    body: Dict[str, Any],
    headers: Dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Result[Any]:
    headers = headers or {"Content-Type": "application/json"}
    try:
        client = await get_http_client()
        resp = await client.request(method, url, json=body, headers=headers)
        if resp.status_code >= 500:
            return err(RuntimeError("API returned status code 500. Please try again later."))
        json_body = None
        try:
            json_body = resp.json()
        except Exception:
            pass
        if resp.status_code >= 400:
            msg = f"API returned status code {resp.status_code}"
            if isinstance(json_body, dict):
                maybe = (
                    json_body.get("error", {}).get("message")
                    if isinstance(json_body.get("error"), dict)
                    else json_body.get("error")
                )
                if maybe:
                    msg += f": {maybe}"
            return err(RuntimeError(msg))
        return ok(json_body)
    except Exception as e:
        return err(e)


