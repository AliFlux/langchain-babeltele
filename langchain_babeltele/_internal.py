"""Small shared helpers. Private module — not part of the public API."""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

#: A chat model, or a string to be resolved via ``init_chat_model``.
ModelLike = str | BaseChatModel

_SCORE_RE = re.compile(r"\d*\.\d+|\d+")


def resolve_model(model: ModelLike) -> BaseChatModel:
    """Coerce a model spec into a concrete chat model (provider-agnostic)."""
    if isinstance(model, str):
        from langchain.chat_models import init_chat_model

        return init_chat_model(model)
    return model


def message_text(content: object) -> str:
    """Flatten message content (string or multimodal block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        ]
        return "".join(parts)
    return str(content)


def render_messages(messages: list[BaseMessage]) -> str:
    """Render a message list as ``role: text`` lines for re-compression."""
    return "\n".join(f"{m.type}: {message_text(m.content)}" for m in messages)


def parse_score(text: str) -> float:
    """Extract the first number from a judge response and clamp it to ``[0, 1]``."""
    match = _SCORE_RE.search(text)
    if match is None:
        return 0.0
    return min(1.0, max(0.0, float(match.group())))
