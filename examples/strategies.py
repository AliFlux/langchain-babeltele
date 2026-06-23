"""Compare BabelTele prompt strategies on the same text.

BabelTele is a family of prompts, not one fixed prompt. The BT_P* strategies trade
off differently between raw density and structural fidelity: some chase the shortest
possible output, others impose symbolic mapping rules that better protect entities,
numbers, and logic. This script runs the same source text through several strategies
and prints a side-by-side comparison so you can feel the difference.

Run it:

    pip install langchain-babeltele python-dotenv langchain-anthropic
    cp examples/default.env examples/.env   # then add your API key
    python examples/strategies.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_babeltele import BabelTeleCompressor, BabelTeleStrategy

# Load default.env first, then .env on top so a local .env can override it.
_HERE = Path(__file__).parent
load_dotenv(_HERE / "default.env")
load_dotenv(_HERE / ".env", override=True)


# A fact-dense paragraph, where the strategies' differing fidelity actually shows.
TEXT = (
    "The Helios-3 satellite launched on March 14, 2026 from Kourou aboard an Ariane 6 "
    "rocket. It carries a 1.2-meter mirror and three instruments: a hyperspectral "
    "imager, a thermal radiometer, and a laser altimeter. Operating in a sun-synchronous "
    "orbit at 705 km, it revisits each point on Earth every 16 days. The mission is "
    "funded jointly by ESA (60 percent) and a consortium of five national agencies "
    "(40 percent), with a total budget of 480 million euros over a 7-year lifetime."
)

# A representative spread: the canonical prompt, two terse variants, and two of the
# structured-mapping variants. Add or swap in any other BabelTeleStrategy members.
STRATEGIES: list[BabelTeleStrategy] = [
    BabelTeleStrategy.DEFAULT,
    BabelTeleStrategy.BT_P3,
    BabelTeleStrategy.BT_P8,
    BabelTeleStrategy.BT_P11,
    BabelTeleStrategy.BT_P13,
]


def main() -> None:
    model = os.environ.get("BABELTELE_MODEL", "anthropic:claude-sonnet-4-6")
    print(f"Using compression model: {model}\n")

    print("ORIGINAL:")
    print(TEXT)

    source_tokens = 0
    summary: list[tuple[str, int, float]] = []
    for strategy in STRATEGIES:
        compressor = BabelTeleCompressor(model, strategy=strategy)
        result = compressor.compress(TEXT)
        source_tokens = result.source_tokens
        summary.append((result.strategy, result.output_tokens, result.retention_ratio))

        print()
        print("=" * 70)
        print(f"{result.strategy}  ->  {result.output_tokens} tokens "
              f"({result.retention_ratio:.0%} of original)")
        print("=" * 70)
        print(result.text)

    # Leaderboard, densest first.
    print()
    print("=" * 70)
    print(f"SUMMARY (source was {source_tokens} tokens)")
    print("=" * 70)
    print(f"  {'strategy':<10}{'tokens':>8}{'retention':>12}")
    for name, tokens, ratio in sorted(summary, key=lambda row: row[2]):
        print(f"  {name:<10}{tokens:>8}{ratio:>11.0%}")


if __name__ == "__main__":
    main()
