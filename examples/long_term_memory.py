"""Long-term memory example: compress past sessions, then retrieve them.

This follows the paper's LoCoMo recipe, wrapped by ``BabelTeleMemoryStore``: each
conversation session is compressed into a dense BabelTele representation, embedded,
and stored. At query time the most relevant compressed sessions come back to ground
a later answer, at a fraction of their original token cost.

The example stores three sessions in an in-memory vector store, then asks a
question that should pull back only the relevant one. It prints each session's
compression and the dense text that gets retrieved.

Run it:

    pip install langchain-babeltele python-dotenv langchain-anthropic langchain-openai
    cp examples/default.env examples/.env   # then add your API keys
    python examples/long_term_memory.py

You need two providers configured: a chat model (BABELTELE_MODEL) for compression
and an embedding model (BABELTELE_EMBEDDINGS) for retrieval.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_babeltele import BabelTeleCompressor, BabelTeleMemoryStore

# BabelTele output is full of emoji and symbols; force UTF-8 so it prints on
# consoles that default to a legacy codepage (e.g. cp1252 on Windows).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load default.env first, then .env on top so a local .env can override it.
_HERE = Path(__file__).parent
load_dotenv(_HERE / "default.env")
load_dotenv(_HERE / ".env", override=True)


# Three past sessions an assistant has had with the same user, each carrying
# decisions worth remembering weeks later.
SESSIONS: list[dict[str, str]] = [
    {
        "date": "2026-03-12",
        "topic": "kickoff",
        "text": (
            "We kicked off the SaaS project today. The user is a solo founder building "
            "a tool that turns meeting recordings into searchable notes. They're "
            "bootstrapped, no outside funding, and want to stay that way for now. "
            "Target customer is small consultancies, 5 to 50 people. They mentioned "
            "they're vegetarian and prefer morning calls, before 11am their time. "
            "The working name for the product is Scribeline."
        ),
    },
    {
        "date": "2026-04-03",
        "topic": "pricing",
        "text": (
            "Long call about pricing. We settled on three tiers: a free plan capped at "
            "5 hours of transcription a month, a Pro plan at 29 dollars a month, and a "
            "Team plan at 99 dollars a month with shared workspaces. Annual billing gets "
            "two months free. We deliberately decided against usage-based pricing for now "
            "because the founder felt unpredictable bills would scare off small "
            "consultancies. We'll revisit metered pricing once there are enterprise leads."
        ),
    },
    {
        "date": "2026-05-20",
        "topic": "hiring",
        "text": (
            "Talked through the first hire. The founder is stretched thin on design, so "
            "the plan is to bring on a product designer before any engineers. Budget is "
            "around 120,000 dollars for the role, remote is fine, and they'd prefer "
            "someone who has shipped a B2B product before. Engineering help comes later, "
            "probably contract first rather than full-time."
        ),
    },
]

QUESTION = "What did we decide about pricing, and why did we skip usage-based billing?"


def main() -> None:
    chat_model = os.environ.get("BABELTELE_MODEL", "anthropic:claude-sonnet-4-6")
    embed_model = os.environ.get("BABELTELE_EMBEDDINGS", "openai:text-embedding-3-small")
    print(f"Compression model : {chat_model}")
    print(f"Embedding model   : {embed_model}\n")

    compressor = BabelTeleCompressor(chat_model)
    store = InMemoryVectorStore(init_embeddings(embed_model))
    memory = BabelTeleMemoryStore(store, compressor)

    print("=" * 70)
    print("STORING SESSIONS (each is compressed before it's embedded)")
    print("=" * 70)
    for session in SESSIONS:
        original_tokens = compressor.count_tokens(session["text"])
        memory.add_session(
            session["text"],
            metadata={
                "date": session["date"],
                "topic": session["topic"],
                "original_tokens": original_tokens,
            },
        )
        print(f"  stored [{session['date']} / {session['topic']}] "
              f"-- {original_tokens} source tokens compressed and embedded")
    print()

    print("=" * 70)
    print(f"QUERY: {QUESTION}")
    print("=" * 70)
    hits = memory.retrieve(QUESTION, k=2)
    print(f"Retrieved {len(hits)} most relevant session(s):\n")

    for rank, doc in enumerate(hits, start=1):
        source_tokens = doc.metadata.get("original_tokens", 0)
        stored_tokens = compressor.count_tokens(doc.page_content)
        ratio = stored_tokens / source_tokens if source_tokens else 1.0
        print("-" * 70)
        print(f"#{rank}  [{doc.metadata.get('date')} / {doc.metadata.get('topic')}]")
        print(f"     stored {source_tokens} -> {stored_tokens} tokens "
              f"({ratio:.0%} of original)")
        print("-" * 70)
        print("Dense BabelTele text that grounds the answer:\n")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()
