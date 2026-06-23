"""Tests for the long-term memory store."""

from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_babeltele import BabelTeleCompressor, BabelTeleMemoryStore

from .conftest import make_model


def _store() -> BabelTeleMemoryStore:
    vector_store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
    engine = BabelTeleCompressor(make_model(["compressed-session"]))
    return BabelTeleMemoryStore(vector_store, engine)


def test_add_session_stores_compressed_text():
    store = _store()
    store.add_session("a long conversation session " * 10, metadata={"session": 1})

    results = store.retrieve("conversation", k=1)
    assert len(results) == 1
    assert results[0].page_content == "compressed-session"
    assert results[0].metadata["session"] == 1


async def test_async_add_and_retrieve():
    store = _store()
    await store.aadd_session("session text " * 10)
    results = await store.aretrieve("session", k=1)
    assert results[0].page_content == "compressed-session"


def test_as_retriever_returns_retriever():
    retriever = _store().as_retriever()
    assert hasattr(retriever, "invoke")
