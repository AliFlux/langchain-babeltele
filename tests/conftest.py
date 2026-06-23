"""Shared fixtures: API-key-free fake models for fast, deterministic tests."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.fixture
def fake_compressor_model() -> FakeListChatModel:
    """A model that always returns a short, dense-looking compression."""
    return FakeListChatModel(responses=["Ent:x|y=>z"])


def make_model(responses: list[str]) -> FakeListChatModel:
    """A fake chat model cycling through ``responses``."""
    return FakeListChatModel(responses=responses)
