from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field
from .types import ModelOptions, FewShotExample

ApiProvider = Literal["openai", "openrouter", "gemini"]


class OpenAISettings(BaseModel):
    key: str = ""
    url: str = "https://api.openai.com/v1/responses"
    model: str = "gpt-4o-mini"


class OpenRouterSettings(BaseModel):
    key: str = ""
    url: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "openai/gpt-4o-mini"
    site_url: str | None = None
    app_title: str | None = None


class GeminiSettings(BaseModel):
    key: str = ""
    model: str = "gemini-2.5-flash"


class Settings(BaseModel):
    api_provider: ApiProvider = "gemini"
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)

    model_options: ModelOptions = Field(default_factory=ModelOptions)
    system_message: str = (
        "You insert text at <mask/> so the combined document reads naturally. "
        "Use BOTH the prefix and suffix as context. Output only the text to insert. "
        "Do not repeat what is already present in the prefix. Avoid reprinting the suffix. "
        "Do not output only whitespace. If unsure, produce a short continuation (2–8 words)."
    )
    user_message_template: str = (
        "Insert text at <mask/> so the final text flows from <prefix/> to <suffix/>.\n"
        "<prefix/>\n{{ prefix }}\n</prefix/>\n"
        "<mask/>\n"
        "<suffix/>\n{{ suffix }}\n</suffix/>\n"
        "Return ONLY the insertion."
    )
    
    _of_thought_removal_regex: str = r"(?!)"  # disabled (never matches)
    def _default_few_shots() -> List[FewShotExample]:  # type: ignore[no-redef]
        return [
            FewShotExample(
                context="Text",
                input="PREFIX: The quick brown <mask/> SUFFIX: over the lazy dog.",
                answer="fox jumps "
            ),
            FewShotExample(
                context="Text",
                input=(
                    "PREFIX: In conclusion, we find that <mask/> SUFFIX: . Therefore, future work should..."
                ),
                answer="the proposed method outperforms baselines by a wide margin"
            ),
        ]

    few_shot_examples: List[FewShotExample] = Field(default_factory=_default_few_shots)

    # Pre/Post settings
    dont_include_dataviews: bool = True
    max_prefix_char_limit: int = 5000
    max_suffix_char_limit: int = 500

    remove_duplicate_math_block_indicator: bool = True
    remove_duplicate_code_block_indicator: bool = True
    debug_mode: bool = False

    # Streaming settings
    enable_streaming: bool = False
    stream_min_chars_before_emit: int = 8
    stream_emit_on_boundary: bool = True
    stream_throttle_ms: int = 40


