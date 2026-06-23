"""Tests for the RAG document compressor."""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_babeltele import BabelTeleCompressor, BabelTeleDocumentCompressor

from .conftest import make_model

DOCS = [
    Document(page_content="A long verbose passage about foxes. " * 10, metadata={"id": 1}),
    Document(page_content="Another verbose passage about dogs. " * 10, metadata={"id": 2}),
]


def _compressor() -> BabelTeleDocumentCompressor:
    engine = BabelTeleCompressor(make_model(["dense!"]))
    return BabelTeleDocumentCompressor(compressor=engine)


def test_compress_documents_rewrites_content_and_keeps_metadata():
    out = _compressor().compress_documents(DOCS, query="foxes?")

    assert [d.page_content for d in out] == ["dense!", "dense!"]
    assert out[0].metadata["id"] == 1
    assert 0 < out[0].metadata["babeltele_retention"] < 1


def test_query_aware_does_not_leak_query_into_output():
    engine = BabelTeleCompressor(make_model(["dense!"]))
    compressor = BabelTeleDocumentCompressor(compressor=engine, query_aware=True)
    out = compressor.compress_documents(DOCS, query="foxes?")
    assert out[0].page_content == "dense!"


async def test_acompress_documents():
    out = await _compressor().acompress_documents(DOCS, query="dogs?")
    assert all(d.page_content == "dense!" for d in out)
