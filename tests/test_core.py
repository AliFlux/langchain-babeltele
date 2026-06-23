"""Tests for the core compression engine."""

from __future__ import annotations

import pytest

from langchain_babeltele import BabelTeleCompressor, BabelTeleStrategy, CompressionResult

from .conftest import make_model

SOURCE = "The quick brown fox jumps over the lazy dog. " * 20


def test_compress_returns_result_with_metrics(fake_compressor_model):
    compressor = BabelTeleCompressor(fake_compressor_model)
    result = compressor.compress(SOURCE)

    assert isinstance(result, CompressionResult)
    assert result.text == "Ent:x|y=>z"
    assert result.source_tokens > result.output_tokens
    assert 0 < result.retention_ratio < 1
    assert result.strategy == "default"
    assert result.verified is None


def test_retention_ratio_handles_empty_source(fake_compressor_model):
    result = BabelTeleCompressor(fake_compressor_model).compress("")
    assert result.retention_ratio == 1.0


def test_custom_prompt_marks_strategy_custom(fake_compressor_model):
    compressor = BabelTeleCompressor(fake_compressor_model, strategy="shrink this: ")
    assert compressor.compress(SOURCE).strategy == "custom"


def test_named_strategy_is_reported(fake_compressor_model):
    compressor = BabelTeleCompressor(fake_compressor_model, strategy=BabelTeleStrategy.BT_P8)
    assert compressor.compress(SOURCE).strategy == "bt_p8"


def test_chunking_splits_and_joins_long_input():
    model = make_model(["AA", "BB", "CC", "DD", "EE"])
    compressor = BabelTeleCompressor(model, chunk_tokens=10)
    text = compressor.compress(SOURCE).text
    # Multiple chunks are joined with a blank line and reuse the cycling responses.
    assert "\n\n" in text


def test_as_runnable_returns_string(fake_compressor_model):
    runnable = BabelTeleCompressor(fake_compressor_model).as_runnable()
    assert runnable.invoke(SOURCE) == "Ent:x|y=>z"


async def test_acompress(fake_compressor_model):
    result = await BabelTeleCompressor(fake_compressor_model).acompress(SOURCE)
    assert result.text == "Ent:x|y=>z"


async def test_as_runnable_async(fake_compressor_model):
    runnable = BabelTeleCompressor(fake_compressor_model).as_runnable()
    assert await runnable.ainvoke(SOURCE) == "Ent:x|y=>z"
