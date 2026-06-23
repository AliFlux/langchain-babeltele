"""Agent integration: BabelTele as LangChain agent middleware.

This middleware inserts BabelTele at the two points where context accumulates in
an agent loop:

* ``wrap_model_call`` — when the running history exceeds a token budget, older
  turns are folded into a single dense message (a denser alternative to the
  built-in ``SummarizationMiddleware``). This also serves as short-term memory
  compression by keeping only the last ``keep_last_n`` turns verbatim.
* ``wrap_tool_call`` — large tool outputs are compressed before they re-enter the
  context, where they are otherwise the biggest token sink in agentic loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ._internal import message_text, render_messages

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware import ModelRequest, ModelResponse, ToolCallRequest
    from langchain_core.messages import BaseMessage
    from langgraph.types import Command

    from .core import BabelTeleCompressor

    ToolResult = ToolMessage | Command

__all__ = ["BabelTeleCompressionMiddleware"]

_DIGEST_PREFIX = "[Compressed prior context]\n"


class BabelTeleCompressionMiddleware(AgentMiddleware):
    """Compress agent history and tool outputs with BabelTele.

    Args:
        compressor: The engine used for compression.
        token_budget: Compress history only when it exceeds this many tokens.
        keep_last_n: Number of most-recent (non-system) messages kept verbatim.
        tool_output_threshold: Compress tool outputs larger than this token count.
    """

    def __init__(
        self,
        compressor: BabelTeleCompressor,
        *,
        token_budget: int = 4000,
        keep_last_n: int = 2,
        tool_output_threshold: int = 2000,
    ) -> None:
        super().__init__()
        self._compressor = compressor
        self._token_budget = token_budget
        self._keep_last_n = keep_last_n
        self._tool_output_threshold = tool_output_threshold

    # -- model calls: history compression ----------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        older = self._overflowing(request)
        if older is not None:
            digest = self._compressor.compress(render_messages(older)).text
            request = request.override(messages=self._rebuild(request.messages, digest))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        older = self._overflowing(request)
        if older is not None:
            digest = (await self._compressor.acompress(render_messages(older))).text
            request = request.override(messages=self._rebuild(request.messages, digest))
        return await handler(request)

    # -- tool calls: output compression ------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolResult],
    ) -> ToolResult:
        result = handler(request)
        if not self._tool_too_big(result):
            return result
        digest = self._compressor.compress(message_text(result.content)).text
        return result.model_copy(update={"content": digest})

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolResult]],
    ) -> ToolResult:
        result = await handler(request)
        if not self._tool_too_big(result):
            return result
        digest = (await self._compressor.acompress(message_text(result.content))).text
        return result.model_copy(update={"content": digest})

    # -- helpers ------------------------------------------------------------

    def _overflowing(self, request: ModelRequest) -> list[BaseMessage] | None:
        """Return the older messages to compress, or ``None`` if within budget."""
        messages = request.messages
        if self._count_messages(messages) <= self._token_budget:
            return None
        body = [m for m in messages if not isinstance(m, SystemMessage)]
        if len(body) <= self._keep_last_n:
            return None
        return body[: -self._keep_last_n]

    def _rebuild(self, messages: list[BaseMessage], digest: str) -> list[BaseMessage]:
        system = [m for m in messages if isinstance(m, SystemMessage)]
        body = [m for m in messages if not isinstance(m, SystemMessage)]
        recent = body[-self._keep_last_n :]
        return [*system, HumanMessage(content=f"{_DIGEST_PREFIX}{digest}"), *recent]

    def _count_messages(self, messages: list[BaseMessage]) -> int:
        return sum(self._compressor.count_tokens(message_text(m.content)) for m in messages)

    def _tool_too_big(self, result: object) -> bool:
        return (
            isinstance(result, ToolMessage)
            and self._compressor.count_tokens(message_text(result.content))
            > self._tool_output_threshold
        )
