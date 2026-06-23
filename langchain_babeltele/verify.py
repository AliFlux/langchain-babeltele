"""Fidelity guardrail for BabelTele compression.

Because BabelTele deliberately abandons human readability, a faulty compression
can silently drop information. :class:`FidelityGuardrail` scores recoverability
with an LLM judge and, on failure, retries with milder structured strategies —
falling back to the original text if recoverability cannot be ensured.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from ._internal import ModelLike, parse_score, resolve_model
from .prompts import BabelTeleStrategy

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = ["FidelityGuardrail"]

# Structured variants that bias toward preserving entities, quantities and logic.
_DEFAULT_FALLBACKS = (
    BabelTeleStrategy.BT_P4,
    BabelTeleStrategy.BT_P8,
    BabelTeleStrategy.BT_P6,
)

_JUDGE_INSTRUCTION = (
    "You judge whether a compressed representation preserves the information of a "
    "source text well enough for another LLM to recover it. Consider entities, "
    "numbers, logic and relations. Reply with ONLY a single number in [0, 1], "
    "where 1 means fully faithful and 0 means information was lost."
)


class FidelityGuardrail:
    """Verify that a compression is recoverable, retrying on failure.

    Args:
        reader: Judge model, or a string resolved via ``init_chat_model``.
        threshold: Minimum fidelity score in ``[0, 1]`` to accept a compression.
        max_retries: Number of fallback strategies to try before giving up.
        fallback_strategies: Strategies attempted in order on failure. Defaults to
            structured variants that better preserve hard facts.
    """

    def __init__(
        self,
        reader: ModelLike,
        *,
        threshold: float = 0.8,
        max_retries: int = 2,
        fallback_strategies: tuple[BabelTeleStrategy, ...] = _DEFAULT_FALLBACKS,
    ) -> None:
        self._reader: BaseChatModel = resolve_model(reader)
        self._threshold = threshold
        self._fallbacks = fallback_strategies[:max_retries]
        self._parser = StrOutputParser()

    def ensure(
        self,
        *,
        source: str,
        compressed: str,
        recompress: Callable[[BabelTeleStrategy], str],
    ) -> tuple[str, bool]:
        """Return ``(text, verified)``, recompressing with fallbacks as needed."""
        if self._score(source, compressed) >= self._threshold:
            return compressed, True
        for strategy in self._fallbacks:
            candidate = recompress(strategy)
            if self._score(source, candidate) >= self._threshold:
                return candidate, True
        return source, False

    async def aensure(
        self,
        *,
        source: str,
        compressed: str,
        recompress: Callable[[BabelTeleStrategy], Awaitable[str]],
    ) -> tuple[str, bool]:
        """Async counterpart of :meth:`ensure`."""
        if await self._ascore(source, compressed) >= self._threshold:
            return compressed, True
        for strategy in self._fallbacks:
            candidate = await recompress(strategy)
            if await self._ascore(source, candidate) >= self._threshold:
                return candidate, True
        return source, False

    # -- scoring ------------------------------------------------------------

    def _score(self, source: str, compressed: str) -> float:
        message = HumanMessage(content=self._judge_text(source, compressed))
        return parse_score(self._parser.invoke(self._reader.invoke([message])))

    async def _ascore(self, source: str, compressed: str) -> float:
        message = HumanMessage(content=self._judge_text(source, compressed))
        return parse_score(self._parser.invoke(await self._reader.ainvoke([message])))

    @staticmethod
    def _judge_text(source: str, compressed: str) -> str:
        return f"{_JUDGE_INSTRUCTION}\n\n[SOURCE]\n{source}\n\n[COMPRESSED]\n{compressed}"
