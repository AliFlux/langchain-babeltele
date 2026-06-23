"""Long-term agent memory backed by BabelTele.

Implements the paper's LoCoMo recipe: each session is compressed into a dense
BabelTele representation, embedded, and stored; at query time the most similar
compressed sessions are retrieved to ground the answer. Wraps any LangChain
``VectorStore``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore

    from .core import BabelTeleCompressor

__all__ = ["BabelTeleMemoryStore"]


class BabelTeleMemoryStore:
    """Store and retrieve compressed session memories.

    Args:
        vector_store: Backing store for embedded, compressed sessions.
        compressor: The engine used to compress each session.
    """

    def __init__(self, vector_store: VectorStore, compressor: BabelTeleCompressor) -> None:
        self._store = vector_store
        self._compressor = compressor

    def add_session(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Compress, embed and store one session; return its id."""
        compressed = self._compressor.compress(text)
        return self._store.add_texts([compressed.text], metadatas=[metadata or {}])[0]

    async def aadd_session(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Async counterpart of :meth:`add_session`."""
        compressed = await self._compressor.acompress(text)
        ids = await self._store.aadd_texts([compressed.text], metadatas=[metadata or {}])
        return ids[0]

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Return the ``k`` most relevant compressed sessions."""
        return self._store.similarity_search(query, k=k)

    async def aretrieve(self, query: str, k: int = 4) -> list[Document]:
        """Async counterpart of :meth:`retrieve`."""
        return await self._store.asimilarity_search(query, k=k)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Expose the underlying store as a retriever for LCEL composition."""
        return self._store.as_retriever(**kwargs)
