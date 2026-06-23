"""langchain-babeltele: model-native high-density text compression for LangChain.

BabelTele compresses verbose text into a dense, non-human-readable but
LLM-recoverable form, then lets downstream models consume it directly. The same
compression engine plugs in at every layer where context accumulates:

* :class:`BabelTeleCompressor` — the core ``str -> str`` engine (and Runnable).
* :class:`BabelTeleDocumentCompressor` — for ``ContextualCompressionRetriever``.
* :class:`BabelTeleCompressionMiddleware` — for ``create_agent`` (history + tools).
* :class:`BabelTeleMemoryStore` — long-term, compressed agent memory.
* :class:`FidelityGuardrail` — recoverability check with fallback retries.

Based on "Large Language Models Do Not Always Need Readable Language"
(arXiv:2606.19857).
"""

from __future__ import annotations

from .compressor import BabelTeleDocumentCompressor
from .core import BabelTeleCompressor
from .memory import BabelTeleMemoryStore
from .middleware import BabelTeleCompressionMiddleware
from .prompts import BabelTeleStrategy, get_prompt
from .result import CompressionResult
from .verify import FidelityGuardrail

__version__ = "0.1.0"

__all__ = [
    "BabelTeleCompressionMiddleware",
    "BabelTeleCompressor",
    "BabelTeleDocumentCompressor",
    "BabelTeleMemoryStore",
    "BabelTeleStrategy",
    "CompressionResult",
    "FidelityGuardrail",
    "__version__",
    "get_prompt",
]
