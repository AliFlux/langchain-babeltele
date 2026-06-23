"""Basic example: compress an agent's conversation history with BabelTele.

When an agent loop runs long, older turns pile up and dominate the token bill.
``BabelTeleCompressionMiddleware`` handles this automatically inside ``create_agent``
by folding the overflowing history into one dense message. This script shows the
operation that middleware performs, in isolation, so you can see the before/after.

Run it:

    pip install langchain-babeltele python-dotenv langchain-anthropic
    cp examples/default.env examples/.env   # then add your API key
    python examples/compress_history.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_babeltele import BabelTeleCompressor

# BabelTele output is full of emoji and symbols; force UTF-8 so it prints on
# consoles that default to a legacy codepage (e.g. cp1252 on Windows).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load default.env first, then .env on top so a local .env can override it.
_HERE = Path(__file__).parent
load_dotenv(_HERE / "default.env")
load_dotenv(_HERE / ".env", override=True)


# A multi-turn conversation that an agent has accumulated. In a real loop these
# would be the older messages the middleware decides to fold away.
CONVERSATION: list[BaseMessage] = [
    HumanMessage(
        "I'm headed to Japan in April and really want to catch the cherry blossoms, "
        "but I'd rather not be shoulder-to-shoulder with tour groups the whole time. "
        "Any thoughts on timing and where to go?"
    ),
    AIMessage(
        "April is peak sakura season, so the famous spots will be packed no matter what. "
        "Tokyo and Kyoto usually bloom from late March into early April. The trick is to "
        "go north. Tohoku peaks a couple of weeks later, around mid-to-late April, and "
        "Hirosaki Castle in particular is worth the trip. Hokkaido runs later still, into "
        "early May, with a fraction of the crowds."
    ),
    HumanMessage(
        "Hirosaki sounds perfect. How long would you stay, and what's the trip out "
        "from Tokyo like?"
    ),
    AIMessage(
        "Two nights is plenty. You'd take the Tohoku Shinkansen from Tokyo to Shin-Aomori, "
        "which is about 3 hours 20 minutes, then hop a local train the last 40 minutes "
        "into Hirosaki. If you're doing other long hauls on the trip, a JR Pass probably "
        "pays for itself."
    ),
    HumanMessage(
        "Good to know. One worry: I'm vegetarian. Am I going to struggle to eat well "
        "up there?"
    ),
    AIMessage(
        "You'll be fine, but it takes some planning. The sneaky one is dashi, the fish "
        "stock that ends up in all sorts of supposedly vegetable dishes, so it's worth "
        "learning how to ask for no fish or meat. Shojin ryori, the Buddhist temple "
        "cuisine, is a safe bet when you can find it, and convenience stores always have "
        "onigiri and tofu to fall back on. Rural towns are trickier than the cities."
    ),
    HumanMessage(
        "Last thing, then I'll stop pestering you: roughly what should I budget per day "
        "for mid-range travel?"
    ),
    AIMessage(
        "Figure somewhere around 15,000 to 20,000 yen a day per person. A business hotel "
        "or a modest ryokan eats up 8,000 to 12,000 of that, food is another 4,000 to "
        "6,000, and the rest covers local trains and entry fees. The JR Pass sits on top "
        "as a separate up-front cost."
    ),
]


def render(messages: list[BaseMessage]) -> str:
    """Render a message list as ``role: text`` lines (what the middleware compresses)."""
    return "\n".join(f"{m.type}: {m.text}" for m in messages)


def main() -> None:
    model = os.environ.get("BABELTELE_MODEL", "anthropic:claude-sonnet-4-6")
    print(f"Using compression model: {model}\n")

    compressor = BabelTeleCompressor(model)
    history = render(CONVERSATION)

    print("=" * 70)
    print(f"ORIGINAL HISTORY ({len(CONVERSATION)} messages)")
    print("=" * 70)
    print(history)
    print()

    print("Compressing history...\n")
    result = compressor.compress(history)

    print("=" * 70)
    print("COMPRESSED (BabelTele dense representation)")
    print("=" * 70)
    print(result.text)
    print()

    saved = result.source_tokens - result.output_tokens
    pct = (1 - result.retention_ratio) * 100
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  strategy        : {result.strategy}")
    print(f"  source tokens   : {result.source_tokens}")
    print(f"  output tokens   : {result.output_tokens}")
    print(f"  retention ratio : {result.retention_ratio:.2%}")
    print(f"  tokens saved    : {saved} ({pct:.1f}% smaller)")


if __name__ == "__main__":
    main()
