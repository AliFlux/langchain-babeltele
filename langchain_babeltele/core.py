"""The BabelTele compression engine.

:class:`BabelTeleCompressor` is the single primitive every higher-level
integration (document compressor, agent middleware, memory store) delegates to.
It performs one model-native compression: a chat-model call driven by a
symbolic-collapse prompt, with optional chunking for over-long inputs and an
optional fidelity guardrail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ._internal import ModelLike, resolve_model
from .prompts import BabelTeleStrategy, get_prompt
from .result import CompressionResult

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from .verify import FidelityGuardrail

__all__ = ["BabelTeleCompressor"]


class BabelTeleCompressor:
    """Compress text into a dense, model-recoverable BabelTele representation.

    Args:
        model: A chat model, or a string resolved via ``init_chat_model``.
        strategy: A built-in :class:`BabelTeleStrategy`, or a raw prompt string
            for a fully custom compression instruction.
        chunk_tokens: When set, inputs longer than this are split into chunks,
            compressed independently, and concatenated (for inputs exceeding the
            compressor's own context window).
        guardrail: Optional :class:`~langchain_babeltele.verify.FidelityGuardrail`
            that verifies recoverability and retries with milder strategies.
    """

    def __init__(
        self,
        model: ModelLike,
        *,
        strategy: BabelTeleStrategy | str = BabelTeleStrategy.DEFAULT,
        chunk_tokens: int | None = None,
        guardrail: FidelityGuardrail | None = None,
    ) -> None:
        self._model = resolve_model(model)
        self._chunk_tokens = chunk_tokens
        self._guardrail = guardrail
        self._parser = StrOutputParser()

        if isinstance(strategy, BabelTeleStrategy):
            self._instruction = get_prompt(strategy)
            self._strategy_name = str(strategy)
        else:
            self._instruction = strategy
            self._strategy_name = "custom"

    # -- public API ---------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Token count under the compressor's model (used for budgeting)."""
        return self._model.get_num_tokens(text)

    def compress(self, text: str) -> CompressionResult:
        """Compress ``text`` synchronously."""
        source_tokens = self.count_tokens(text)
        compressed = (
            self._compress_chunked(text)
            if self._should_chunk(source_tokens)
            else self._compress_one(text)
        )
        verified: bool | None = None
        if self._guardrail is not None:
            compressed, verified = self._guardrail.ensure(
                source=text,
                compressed=compressed,
                recompress=lambda s: self._compress_one(text, instruction=get_prompt(s)),
            )
        return self._result(text, compressed, source_tokens, verified)

    async def acompress(self, text: str) -> CompressionResult:
        """Compress ``text`` asynchronously."""
        source_tokens = self.count_tokens(text)
        compressed = (
            await self._acompress_chunked(text)
            if self._should_chunk(source_tokens)
            else await self._acompress_one(text)
        )
        verified: bool | None = None
        if self._guardrail is not None:
            compressed, verified = await self._guardrail.aensure(
                source=text,
                compressed=compressed,
                recompress=lambda s: self._acompress_one(text, instruction=get_prompt(s)),
            )
        return self._result(text, compressed, source_tokens, verified)

    def as_runnable(self) -> Runnable[str, str]:
        """Expose compression as a ``str -> str`` Runnable for LCEL pipelines."""
        return RunnableLambda(
            lambda text: self.compress(text).text,
            afunc=self._acompress_to_text,
        )

    # -- single-shot compression -------------------------------------------

    def _compress_one(self, text: str, *, instruction: str | None = None) -> str:
        messages = self._build(text, instruction)
        return self._parser.invoke(self._model.invoke(messages))

    async def _acompress_one(self, text: str, *, instruction: str | None = None) -> str:
        messages = self._build(text, instruction)
        return self._parser.invoke(await self._model.ainvoke(messages))

    # -- chunked compression -----------------------------------------------

    def _compress_chunked(self, text: str) -> str:
        batches = [self._build(chunk) for chunk in self._split(text)]
        return "\n\n".join(self._parser.invoke(r) for r in self._model.batch(batches))

    async def _acompress_chunked(self, text: str) -> str:
        batches = [self._build(chunk) for chunk in self._split(text)]
        return "\n\n".join(self._parser.invoke(r) for r in await self._model.abatch(batches))

    # -- helpers ------------------------------------------------------------

    def _should_chunk(self, source_tokens: int) -> bool:
        return self._chunk_tokens is not None and source_tokens > self._chunk_tokens

    def _build(self, source: str, instruction: str | None = None) -> list[BaseMessage]:
        return [
            SystemMessage(content=instruction or self._instruction),
            HumanMessage(content=source),
        ]

    def _split(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_tokens,
            chunk_overlap=0,
            length_function=self.count_tokens,
        )
        return splitter.split_text(text)

    async def _acompress_to_text(self, text: str) -> str:
        return (await self.acompress(text)).text

    def _result(
        self, source: str, compressed: str, source_tokens: int, verified: bool | None
    ) -> CompressionResult:
        return CompressionResult(
            text=compressed,
            source_tokens=source_tokens,
            output_tokens=self.count_tokens(compressed),
            strategy=self._strategy_name,
            verified=verified,
        )
