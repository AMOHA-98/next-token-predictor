from __future__ import annotations
import re
from typing import List
from jinja2 import Environment, BaseLoader, StrictUndefined

from ..types import PredictionService, ChatMessage
from ..settings import Settings
from ..utils import Result, ok, err
from ..context_detection import get_context, Context

from ..pre_processors.dataview_remover import DataViewRemover
from ..pre_processors.length_limiter import LengthLimiter
from ..post_processors.remove_code_indicators import RemoveCodeIndicators
from ..post_processors.remove_math_indicators import RemoveMathIndicators
from ..post_processors.remove_overlap import RemoveOverlap
from ..post_processors.remove_whitespace import RemoveWhitespace

from ..api_clients.openai_client import OpenAIClient
from ..api_clients.openrouter_client import OpenRouterClient
from ..api_clients.gemini_client import GeminiClient


class InlineAutoCompleter(PredictionService):
    def __init__(
        self,
        client,
        system_message: str,
        user_message_template: str,
        chain_of_thought_regex: str,
        pre_processors,
        post_processors,
        few_shot_examples,
        debug_mode: bool,
    ):
        self.client = client
        self.system_message = system_message
        self.template = Environment(
            loader=BaseLoader(), autoescape=False, undefined=StrictUndefined
        ).from_string(user_message_template)
        self.remove_regex = re.compile(chain_of_thought_regex, re.M)
        self.pre_processors = pre_processors
        self.post_processors = post_processors
        self.few_shot_examples = few_shot_examples
        self.debug_mode = debug_mode

    @classmethod
    def from_settings(cls, s: Settings) -> "InlineAutoCompleter":
        pre = []
        if s.dont_include_dataviews:
            pre.append(DataViewRemover())
        pre.append(LengthLimiter(s.max_prefix_char_limit, s.max_suffix_char_limit))

        post = []
        if s.remove_duplicate_math_block_indicator: post.append(RemoveMathIndicators())
        if s.remove_duplicate_code_block_indicator: post.append(RemoveCodeIndicators())
        post.extend([RemoveOverlap(), RemoveWhitespace()])

        if s.api_provider == "openai":
            client = OpenAIClient.from_settings(s)
        elif s.api_provider == "openrouter":
            client = OpenRouterClient.from_settings(s)
        elif s.api_provider == "gemini":
            client = GeminiClient.from_settings(s)
        else:
            raise ValueError("Invalid API provider")

        return cls(
            client=client,
            system_message=s.system_message,
            user_message_template=s.user_message_template,
            chain_of_thought_regex=s.chain_of_thought_removal_regex,
            pre_processors=pre,
            post_processors=post,
            few_shot_examples=s.few_shot_examples,
            debug_mode=s.debug_mode,
        )

    async def fetch_predictions(self, prefix: str, suffix: str) -> Result[str]:
        messages = self.build_messages(prefix, suffix)
        if len(messages) == 0:
            return ok("")

        if self.debug_mode:
            print("InlineAutoCompleter messages:\n", [m.model_dump() for m in messages])

        result = await self.client.query_chat_model(messages)

        if self.debug_mode and result.is_ok():
            print("InlineAutoCompleter raw response:\n", result.value)

        result = self._extract_answer(result)

        # Ensure we still have context for post-processing
        context = get_context(prefix, suffix)
        for post in self.post_processors:
            result = result.map(lambda r: post.process(prefix, suffix, r, context))

        return self._guardrails(result)

    def build_messages(self, prefix: str, suffix: str) -> List[ChatMessage]:
        context = get_context(prefix, suffix)

        for p in self.pre_processors:
            if getattr(p, "removes_cursor")(prefix, suffix):
                return []
            res = p.process(prefix, suffix, context)
            prefix, suffix = res.prefix, res.suffix

        few_shot_msgs: List[ChatMessage] = []
        for ex in self.few_shot_examples:
            if ex.context == context.value:
                few_shot_msgs.extend([
                    ChatMessage(role="user", content=ex.input),
                    ChatMessage(role="assistant", content=ex.answer),
                ])

        messages = [
            ChatMessage(role="system", content=self._system_for(context)),
            *few_shot_msgs,
            ChatMessage(role="user", content=self.template.render(prefix=prefix, suffix=suffix)),
        ]
        return messages

    def _system_for(self, context: Context) -> str:
        base = self.system_message
        if context == Context.Text:
            return base + "\n\nThe <mask/> is in a paragraph; complete it naturally in the same language without overlap."
        if context == Context.Heading:
            return base + "\n\nThe <mask/> is in a heading; complete the title to fit the content."
        if context == Context.BlockQuotes:
            return base + "\n\nThe <mask/> is within a quote; complete it to fit the context."
        if context == Context.UnorderedList:
            return base + "\n\nThe <mask/> is in an unordered list; add item(s) that fit, no overlap."
        if context == Context.NumberedList:
            return base + "\n\nThe <mask/> is in a numbered list; add item(s) that fit sequence/context."
        if context == Context.CodeBlock:
            return base + "\n\nThe <mask/> is in a code block; complete in the same language and support surrounding text."
        if context == Context.MathBlock:
            return base + "\n\nThe <mask/> is in a math block; output only LaTeX (no prose)."
        if context == Context.TaskList:
            return base + "\n\nThe <mask/> is in a task list; add logical (sub)tasks."
        return base

    def _extract_answer(self, result: Result[str]) -> Result[str]:
        if result.is_err():
            return result
        text = result.value
        if not self.remove_regex.search(text):
            return ok(text)
        return ok(self.remove_regex.sub("", text))

    def _guardrails(self, result: Result[str]) -> Result[str]:
        if result.is_err(): return result
        v = result.value or ""
        if not v.strip():
            return err(RuntimeError("Empty result"))
        if "<mask/>" in v:
            return err(RuntimeError("Mask in result"))
        return ok(v)


