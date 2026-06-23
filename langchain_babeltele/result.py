"""The result object returned by a BabelTele compression."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CompressionResult"]


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """Outcome of compressing one piece of text.

    Attributes:
        text: The compressed BabelTele representation.
        source_tokens: Token count of the original text.
        output_tokens: Token count of ``text``.
        strategy: Name of the prompt strategy used (or ``"custom"``).
        verified: Fidelity-guardrail result — ``True``/``False`` when a guardrail
            ran, or ``None`` when no guardrail was attached.
    """

    text: str
    source_tokens: int
    output_tokens: int
    strategy: str
    verified: bool | None = None

    @property
    def retention_ratio(self) -> float:
        """Compressed length as a fraction of the original (lower is denser)."""
        if self.source_tokens == 0:
            return 1.0
        return self.output_tokens / self.source_tokens
