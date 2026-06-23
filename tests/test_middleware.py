"""Tests for the agent middleware.

The middleware hooks receive ``ModelRequest`` / ``ToolCallRequest`` objects. We
exercise the hooks with lightweight stubs that honour the parts of the contract
the middleware actually uses (``messages`` + ``override`` for model calls; the
handler return value for tool calls), avoiding a full agent graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_babeltele import BabelTeleCompressionMiddleware, BabelTeleCompressor

from .conftest import make_model


@dataclass
class StubRequest:
    """Minimal stand-in for ModelRequest."""

    messages: list

    def override(self, *, messages):
        return replace(self, messages=messages)


def _middleware(**kwargs) -> BabelTeleCompressionMiddleware:
    engine = BabelTeleCompressor(make_model(["DIGEST"]))
    return BabelTeleCompressionMiddleware(engine, **kwargs)


def test_history_under_budget_is_untouched():
    mw = _middleware(token_budget=10_000)
    request = StubRequest(messages=[HumanMessage(content="hi"), AIMessage(content="hello")])
    seen = {}
    mw.wrap_model_call(request, lambda req: seen.setdefault("req", req))
    assert seen["req"] is request


def test_history_over_budget_is_compressed():
    mw = _middleware(token_budget=5, keep_last_n=1)
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="old turn one " * 20),
        AIMessage(content="old turn two " * 20),
        HumanMessage(content="latest question"),
    ]
    captured = {}
    mw.wrap_model_call(StubRequest(messages=messages), lambda req: captured.setdefault("req", req))

    out = captured["req"].messages
    assert isinstance(out[0], SystemMessage)  # system preserved
    assert out[1].content.startswith("[Compressed prior context]")
    assert "DIGEST" in out[1].content
    assert out[-1].content == "latest question"  # last turn kept verbatim


def test_small_tool_output_passes_through():
    mw = _middleware(tool_output_threshold=10_000)
    msg = ToolMessage(content="short", tool_call_id="1")
    assert mw.wrap_tool_call(None, lambda req: msg) is msg


def test_large_tool_output_is_compressed():
    mw = _middleware(tool_output_threshold=5)
    msg = ToolMessage(content="verbose tool output " * 50, tool_call_id="1")
    out = mw.wrap_tool_call(None, lambda req: msg)
    assert out.content == "DIGEST"
    assert out.tool_call_id == "1"  # identity preserved


async def test_async_tool_output_is_compressed():
    mw = _middleware(tool_output_threshold=5)
    msg = ToolMessage(content="verbose tool output " * 50, tool_call_id="1")

    async def handler(req):
        return msg

    out = await mw.awrap_tool_call(None, handler)
    assert out.content == "DIGEST"
