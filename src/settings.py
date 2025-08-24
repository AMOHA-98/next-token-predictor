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
    system_message: str = "You autocomplete text for writers. Output only the completion."
    user_message_template: str = "{{ prefix }}<mask/>{{ suffix }}"
    chain_of_thought_removal_regex: str = r"(?s).*?(?=<final_answer>)|</final_answer>.*"
    few_shot_examples: List[FewShotExample] = Field(default_factory=list)

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


