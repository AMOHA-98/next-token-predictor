from __future__ import annotations
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from typing import Any
import asyncio
from pydantic import BaseModel
from .settings import Settings
from .prediction_services.inline_autocomplete import InlineAutoCompleter
from .utils import init_http_client, close_http_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_http_client()
    try:
        yield
    finally:
        await close_http_client()


app = FastAPI(title="Autocomplete Prediction Service", lifespan=lifespan)

# You can load from env or a file; here we use defaults for brevity.
settings = Settings()
service = InlineAutoCompleter.from_settings(settings)


class PredictRequest(BaseModel):
    prefix: str
    suffix: str


class PredictResponse(BaseModel):
    completion: str


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    res = await service.fetch_predictions(req.prefix, req.suffix)
    if res.is_ok():
        return PredictResponse(completion=res.value)
    # If you prefer HTTP error codes, raise HTTPException here instead.
    return PredictResponse(completion="")


@app.post("/predict/stream")
async def predict_stream(req: PredictRequest):
    if not settings.enable_streaming:
        # Fall back to non-streaming predict
        res = await service.fetch_predictions(req.prefix, req.suffix)
        text = res.value if res.is_ok() else ""
        async def one_shot():
            yield text
        return StreamingResponse(one_shot(), media_type="text/event-stream")

    async def event_gen():
        messages = service.build_messages(req.prefix, req.suffix)
        if not messages:
            return
        buffer = ""
        last_emit = 0.0
        min_chars = settings.stream_min_chars_before_emit
        throttle = settings.stream_throttle_ms / 1000.0
        boundary = settings.stream_emit_on_boundary
        def is_boundary(text: str) -> bool:
            if not text:
                return False
            return text[-1].isspace() or text[-1] in ".,;:!?)]}\"'"
        async for chunk in service.client.stream_chat_model(messages):
            if not isinstance(chunk, str) or not chunk:
                continue
            buffer += chunk
            now = asyncio.get_event_loop().time()
            should_emit = len(buffer) >= min_chars and (not boundary or is_boundary(buffer)) and (now - last_emit) >= throttle
            if should_emit:
                yield buffer
                last_emit = now
        if buffer:
            # final flush
            yield buffer

    return StreamingResponse(event_gen(), media_type="text/event-stream")


class HealthResponse(BaseModel):
    status: str


class ConfigResponse(BaseModel):
    api_provider: str
    model: str
    url: str
    streaming: bool
    stream_min_chars_before_emit: int
    stream_emit_on_boundary: bool
    stream_throttle_ms: int
    model_options: dict[str, Any]


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.get("/config", response_model=ConfigResponse)
async def config():
    if settings.api_provider == "openai":
        model = settings.openai.model
        url = settings.openai.url
    else:
        model = settings.openrouter.model
        url = settings.openrouter.url
    return ConfigResponse(
        api_provider=settings.api_provider,
        model=model,
        url=url,
        streaming=settings.enable_streaming,
        stream_min_chars_before_emit=settings.stream_min_chars_before_emit,
        stream_emit_on_boundary=settings.stream_emit_on_boundary,
        stream_throttle_ms=settings.stream_throttle_ms,
        model_options=settings.model_options.model_dump(),
    )

