"""Guard compression fidelity with the FidelityGuardrail.

Because BabelTele abandons readability, a bad compression can silently drop facts.
The guardrail scores recoverability with an LLM judge and, if the score falls below
a threshold, retries with milder structured strategies before falling back to the
original text. The CompressionResult carries a ``verified`` flag recording what
happened.

Run it:

    pip install langchain-babeltele python-dotenv langchain-anthropic
    cp examples/default.env examples/.env   # then add your API key
    python examples/guardrail.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_babeltele import BabelTeleCompressor, FidelityGuardrail

# BabelTele output is full of emoji and symbols; force UTF-8 so it prints on
# consoles that default to a legacy codepage (e.g. cp1252 on Windows).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load default.env first, then .env on top so a local .env can override it.
_HERE = Path(__file__).parent
load_dotenv(_HERE / "default.env")
load_dotenv(_HERE / ".env", override=True)


# Fact-heavy text: exactly the kind of input where a sloppy compression would lose
# a number or misbind an entity, and where the guardrail earns its keep.
TEXT = (
    "Patient is a 54-year-old male, 82 kg, presenting with a systolic blood pressure "
    "of 168 mmHg and an LDL of 190 mg/dL. Started on atorvastatin 40 mg nightly and "
    "lisinopril 10 mg daily. Allergic to penicillin (rash). Follow-up labs in 6 weeks; "
    "if LDL remains above 130, increase atorvastatin to 80 mg. Do not combine with "
    "the gemfibrozil he took in 2024."
)


def main() -> None:
    model = os.environ.get("BABELTELE_MODEL", "anthropic:claude-sonnet-4-6")
    print(f"Using compression and judge model: {model}\n")

    # The judge can be any model; here it reuses the same one. Raise the threshold
    # (e.g. 0.95) to make the guardrail stricter and more likely to retry/fall back.
    guardrail = FidelityGuardrail(model, threshold=0.8)
    compressor = BabelTeleCompressor(model, guardrail=guardrail)

    print("ORIGINAL:")
    print(TEXT)
    print()

    print("Compressing with fidelity guardrail...\n")
    result = compressor.compress(TEXT)

    print("COMPRESSED:")
    print(result.text)
    print()

    verdict = {
        True: "passed the fidelity check (or a fallback strategy did)",
        False: "failed every attempt -> fell back to the ORIGINAL text",
        None: "no guardrail ran",
    }[result.verified]

    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  strategy        : {result.strategy}")
    print(f"  source tokens   : {result.source_tokens}")
    print(f"  output tokens   : {result.output_tokens}")
    print(f"  retention ratio : {result.retention_ratio:.0%}")
    print(f"  verified        : {result.verified} ({verdict})")
    if result.verified is False:
        print("\n  Note: output == original here, so retention is 100% by design.")
        print("  The guardrail chose safety over density.")


if __name__ == "__main__":
    main()
