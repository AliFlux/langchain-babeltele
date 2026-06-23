"""RAG integration: a ``BaseDocumentCompressor`` backed by BabelTele.

Drop into a ``ContextualCompressionRetriever`` to shrink retrieved documents into
dense, model-recoverable form before they enter the prompt. Task-agnostic by
default (the query is ignored, matching the paper); set ``query_aware=True`` to
focus the compression on the query.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict

from .core import BabelTeleCompressor
from .result import CompressionResult

if TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

__all__ = ["BabelTeleDocumentCompressor"]


class BabelTeleDocumentCompressor(BaseDocumentCompressor):
    """Compress retrieved documents with a :class:`BabelTeleCompressor`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    compressor: BabelTeleCompressor
    """The engine used to compress each document's content."""

    query_aware: bool = False
    """When ``True``, append the query so compression can focus on it."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str = "",
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        return [
            self._rewrite(doc, self.compressor.compress(self._source(doc, query)))
            for doc in documents
        ]

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str = "",
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        return [
            self._rewrite(doc, await self.compressor.acompress(self._source(doc, query)))
            for doc in documents
        ]

    # -- helpers ------------------------------------------------------------

    def _source(self, doc: Document, query: str) -> str:
        if self.query_aware and query:
            return f"{doc.page_content}\n\n[Focus query]: {query}"
        return doc.page_content

    @staticmethod
    def _rewrite(doc: Document, result: CompressionResult) -> Document:
        return Document(
            page_content=result.text,
            metadata={**doc.metadata, "babeltele_retention": result.retention_ratio},
        )
