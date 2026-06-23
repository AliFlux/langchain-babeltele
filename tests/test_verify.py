"""Tests for the fidelity guardrail."""

from __future__ import annotations

from langchain_babeltele import BabelTeleCompressor, BabelTeleStrategy, FidelityGuardrail

from .conftest import make_model


def test_passes_when_score_above_threshold():
    guardrail = FidelityGuardrail(make_model(["0.95"]), threshold=0.8)
    text, verified = guardrail.ensure(
        source="src", compressed="dense", recompress=lambda s: "should-not-run"
    )
    assert (text, verified) == ("dense", True)


def test_retries_with_fallback_until_recoverable():
    # First judge call fails (0.2), second (the fallback) passes (0.9).
    guardrail = FidelityGuardrail(make_model(["0.2", "0.9"]), threshold=0.8, max_retries=2)
    used: list[BabelTeleStrategy] = []

    def recompress(strategy):
        used.append(strategy)
        return f"fallback:{strategy}"

    text, verified = guardrail.ensure(source="src", compressed="weak", recompress=recompress)

    assert verified is True
    assert text.startswith("fallback:")
    assert len(used) == 1  # stopped after the first successful fallback


def test_falls_back_to_source_when_unrecoverable():
    guardrail = FidelityGuardrail(make_model(["0.1"]), threshold=0.8, max_retries=2)
    text, verified = guardrail.ensure(
        source="original", compressed="weak", recompress=lambda s: "still-weak"
    )
    assert (text, verified) == ("original", False)


def test_guardrail_integrates_with_compressor():
    engine = BabelTeleCompressor(
        make_model(["compressed"]),
        guardrail=FidelityGuardrail(make_model(["0.99"]), threshold=0.8),
    )
    result = engine.compress("a verbose source passage " * 10)
    assert result.verified is True
