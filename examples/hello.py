"""Hello, BabelTele: the smallest possible compression example.

Take one verbose paragraph, compress it into the dense BabelTele representation,
and print the before/after with token counts. Start here.

Run it:

    pip install langchain-babeltele python-dotenv langchain-anthropic
    cp examples/default.env examples/.env   # then add your API key
    python examples/hello.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_babeltele import BabelTeleCompressor

# Load default.env first, then .env on top so a local .env can override it.
_HERE = Path(__file__).parent
load_dotenv(_HERE / "default.env")
load_dotenv(_HERE / ".env", override=True)


TEXT = (
    "Honestly, the meeting could have been an email. We spent the better part of an "
    "hour going in circles about whether to move the launch date, and in the end we "
    "landed exactly where we started: we're keeping the original date, but we'll cut "
    "the analytics dashboard from the first release and ship it a couple of weeks "
    "later as a fast follow."
)


def main() -> None:
    model = os.environ.get("BABELTELE_MODEL", "anthropic:claude-sonnet-4-6")
    print(f"Using compression model: {model}\n")

    compressor = BabelTeleCompressor(model)

    print("ORIGINAL:")
    print(TEXT)
    print()

    result = compressor.compress(TEXT)

    print("COMPRESSED:")
    print(result.text)
    print()

    print(f"{result.source_tokens} -> {result.output_tokens} tokens "
          f"({result.retention_ratio:.0%} of the original)")


if __name__ == "__main__":
    main()
