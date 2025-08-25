from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any
import asyncio
from pydantic import BaseModel
import logging
from .settings import Settings
from .prediction_services.inline_autocomplete import InlineAutoCompleter
from .utils import init_http_client, close_http_client, stable_hash
from fastapi.responses import HTMLResponse
from fastapi import Request

from .singleflight import SingleFlight
from .latest_only import LatestOnly
from .rate_limit import limiter_for
from .cache import suggest_cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_http_client()
    try:
        yield
    finally:
        await close_http_client()


app = FastAPI(title="Autocomplete Prediction Service", lifespan=lifespan)
logger = logging.getLogger("ntp")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger.setLevel(logging.INFO)

# Enable simple, permissive CORS for local testing UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# You can load from env or a file; here we use defaults for brevity.
settings = Settings()
service = InlineAutoCompleter.from_settings(settings)
_sf = SingleFlight()
_latest_only = LatestOnly()


class PredictRequest(BaseModel):
    prefix: str
    suffix: str


class PredictResponse(BaseModel):
    completion: str


def _req_key(provider: str, model: str, prefix: str, suffix: str) -> str:
    tail = prefix[-200:]
    head = suffix[:60]
    return f"{provider}:{model}:{stable_hash(tail+'\u241f'+head)}"  # U+241F SYMBOL FOR UNIT SEPARATOR


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    user = request.headers.get("X-Client-Id") or (request.client.host if request.client else "anon")
    provider = settings.api_provider
    if provider == "openai":
        model = settings.openai.model
    elif provider == "openrouter":
        model = settings.openrouter.model
    elif provider == "gemini":
        model = settings.gemini.model
    else:
        model = "unknown"

    async with limiter_for(user):
        key = _req_key(provider, model, req.prefix, req.suffix)
        if logger.isEnabledFor(logging.INFO):
            logger.info("/predict user=%s key=%s tail=%d head=%d", user, key, len(req.prefix[-200:]), len(req.suffix[:60]))

        cached = None
        try:
            cached = suggest_cache.get(key)  # type: ignore[attr-defined]
        except Exception:
            # cache stub may not support .get in the same way; best-effort
            cached = suggest_cache[key] if key in suggest_cache else None  # type: ignore[index]
        if cached is not None and isinstance(cached, str) and cached != "":
            if logger.isEnabledFor(logging.INFO):
                logger.info("cache hit user=%s key=%s", user, key)
            return PredictResponse(completion=cached)

        async def runner(p: str, s: str) -> str:
            k = _req_key(provider, model, p, s)
            if logger.isEnabledFor(logging.INFO):
                logger.info("latest-only run user=%s key=%s", user, k)

            async def call():
                # hard timeout per call; avoids hung upstreams dragging UI responsiveness
                return await asyncio.wait_for(service.fetch_predictions(p, s), timeout=12.0)

            res = await _sf.do(k, call)
            text = res.value if hasattr(res, "is_ok") and res.is_ok() else ""
            if logger.isEnabledFor(logging.INFO):
                if getattr(res, "is_ok", lambda: False)():
                    logger.info("api ok user=%s key=%s len=%d", user, k, len(text))
                else:
                    logger.info("api err user=%s key=%s err=%s", user, k, getattr(res, "error", None))
            # cache set best-effort
            if text:
                try:
                    suggest_cache[k] = text  # type: ignore[index]
                except Exception:
                    pass
            return text

        text = await _latest_only.run(user, req.prefix, req.suffix, runner)
        if logger.isEnabledFor(logging.INFO):
            logger.info("/predict done user=%s key=%s len=%d", user, key, len(text))
        return PredictResponse(completion=text)


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
    elif settings.api_provider == "openrouter":
        model = settings.openrouter.model
        url = settings.openrouter.url
    elif settings.api_provider == "gemini":
        model = settings.gemini.model
        url = "google-genai"
    else:
        model = "unknown"
        url = ""
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


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    html = r"""<!doctype html>
<meta charset="utf-8" />
<title>Inline Autocomplete — Smoke Test</title>
<style>
  body { font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; margin: 2rem; }
  .wrap { position: relative; width: 780px; }
  textarea, .ghost {
    width: 100%; min-height: 200px; padding: 14px; border-radius: 12px; box-sizing: border-box;
    border: 1px solid #d0d7de; font: inherit; white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .ghost {
    position: absolute; top: 0; left: 0; color: #9aa3ad; pointer-events: none; z-index: 0;
  }
  textarea { background: transparent; position: relative; z-index: 1; }
  .meta { margin-top: .5rem; color: #6b7280; }
</style>
<div class="wrap">
  <div id="ghost" class="ghost"></div>
  <textarea id="box" spellcheck="false" placeholder="Start typing… then pause. Press Tab to accept."></textarea>
</div>
<div class="meta" id="meta"></div>
<script>
const box = document.getElementById('box');
const ghost = document.getElementById('ghost');
const meta  = document.getElementById('meta');

// Stable per-client id for request de-dupe and LatestOnly isolation
const CLIENT_ID_KEY = 'autocompleteClientId';
const clientId = localStorage.getItem(CLIENT_ID_KEY) || (() => {
  const id = (crypto.randomUUID && crypto.randomUUID()) || (Date.now() + '-' + Math.random().toString(16).slice(2));
  localStorage.setItem(CLIENT_ID_KEY, id);
  return id;
})();

// Client policy: adaptive debounce + throttle + cancel + dedupe + boundary
let lastSuggest = '', lastLatency = 0;
let pendingCtrl = null;
let lastKeySig = '';
let lastTriggeredAt = 0;
const MAX_RPS = 3;
let lastRequestPrefixLen = 0;

let lastKeyAt = 0;
const hist = [];
const BOUNDARY = /[\s.,;:!?\-\)\]\}\u00BB\u201D]$/;

function keySig(prefix, suffix) {
  const tail = prefix.slice(-200);
  const head = suffix.slice(0, 60);
  return tail + '\u241f' + head;
}

function currentSig() {
  const i = box.selectionStart;
  return keySig(box.value.slice(0, i), box.value.slice(i));
}

function adaptiveDebounceMs() {
  const intervals = hist.slice().sort((a,b)=>a-b);
  const median = intervals[Math.floor(intervals.length/2)] || 160;
  const ms = Math.round(median * 0.6);
  return Math.max(90, Math.min(ms, 200));
}

function shouldFire(prefix, sinceLastChars) {
  const idleMs = performance.now() - lastKeyAt;
  return BOUNDARY.test(prefix) || sinceLastChars >= 1 || idleMs >= 160;
}

function splitAtCaret(el) {
  const i = el.selectionStart;
  return [el.value.slice(0, i), el.value.slice(i)];
}

function renderGhost() {
  const i = box.selectionStart;
  const before = box.value.slice(0, i);
  const after  = box.value.slice(i);
  ghost.textContent = before + (lastSuggest || '') + after;
}

function syncScroll() {
  ghost.style.transform = `translateY(${-box.scrollTop}px)`;
}

function onSelectionChange() {
  renderGhost();
}

async function fetchSuggest(prefix, suffix, sig) {
  const now = performance.now();
  const minInterval = 1000 / MAX_RPS;
  if (now - lastTriggeredAt < minInterval) {
    const wait = Math.max(0, minInterval - (now - lastTriggeredAt));
    clearTimeout(t);
    t = setTimeout(() => fetchSuggest(prefix, suffix, sig), wait);
    return;
  }
  lastTriggeredAt = performance.now();

  if (pendingCtrl && typeof pendingCtrl.abort === 'function') pendingCtrl.abort();
  pendingCtrl = new AbortController();

  const started = performance.now();
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Client-Id': clientId},
      signal: pendingCtrl.signal,
      body: JSON.stringify({prefix, suffix}),
    });
    if (!res.ok) {
      meta.textContent = `Request failed: ${res.status}`;
      return;
    }
    const json = await res.json().catch(() => ({}));
    lastLatency = Math.round(performance.now() - started);
    lastKeySig = sig;
    lastRequestPrefixLen = prefix.length;
    lastSuggest = (json && json.completion) ? json.completion : '';
    renderGhost();
    meta.textContent = `Latency: ${lastLatency} ms  •  Tokens max: server-side`;
  } catch (e) {
    if (pendingCtrl && pendingCtrl.signal && pendingCtrl.signal.aborted) return;
    lastSuggest = '';
    renderGhost();
    meta.textContent = 'Error fetching suggestion';
  }
}

let t;
function scheduleSuggest() {
  clearTimeout(t);
  lastSuggest = '';
  renderGhost();

  const i = box.selectionStart;
  const j = box.selectionEnd;
  if (i !== j) return;

  const [prefix, suffix] = splitAtCaret(box);
  const boundary = BOUNDARY.test(prefix);
  const sinceLast = Math.max(0, prefix.length - lastRequestPrefixLen);
  const sig = keySig(prefix, suffix);
  const delay = boundary ? 0 : adaptiveDebounceMs();

  const caretAtSchedule = i;
  t = setTimeout(() => {
    if (box.selectionStart !== caretAtSchedule || box.selectionEnd !== caretAtSchedule) return;
    if (!shouldFire(prefix, sinceLast)) return;
    if (sig === lastKeySig) return;
    fetchSuggest(prefix, suffix, sig);
  }, delay);
}

box.addEventListener('input', scheduleSuggest);
box.addEventListener('click', renderGhost);
box.addEventListener('keyup', renderGhost);
box.addEventListener('scroll', syncScroll);
box.addEventListener('select', onSelectionChange);
box.addEventListener('mouseup', onSelectionChange);
box.addEventListener('touchend', onSelectionChange);

box.addEventListener('keydown', (e) => {
  const now = performance.now();
  if (lastKeyAt) {
    hist.push(now - lastKeyAt);
    if (hist.length > 24) hist.shift();
  }
  lastKeyAt = now;

  if (e.key === 'Tab' && lastSuggest) {
    if (currentSig() !== lastKeySig) {
      e.preventDefault();
      lastSuggest = '';
      renderGhost();
      return;
    }
    e.preventDefault();
    const i = box.selectionStart;
    const before = box.value.slice(0, i);
    const after  = box.value.slice(i);
    function longestOverlapEndVsStart(a, b) { // a: completion, b: suffix
      const max = Math.min(a.length, b.length);
      for (let k = max; k > 0; k--) {
        if (a.slice(-k) === b.slice(0, k)) return k;
      }
      return 0;
    }
    const comp = lastSuggest;
    const k = longestOverlapEndVsStart(comp, after);
    box.value = before + comp + after.slice(k);
    const newI = before.length + comp.length;
    box.setSelectionRange(newI, newI);
    lastSuggest = '';
    renderGhost();
  }
});

syncScroll();
renderGhost();
</script>"""
    return HTMLResponse(content=html)

