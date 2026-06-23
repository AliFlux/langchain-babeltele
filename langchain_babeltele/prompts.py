"""The BabelTele prompt family.

Each prompt instructs a model to compress verbose text into a dense,
non-human-readable but LLM-recoverable form by relaxing the readability prior.
They embody the paper's three principles: *omnilingual lexical selection*,
*symbolic collapse*, and *recoverable semantic density*.

Prompts are transcribed from the paper's Appendix C (arXiv:2606.19857). ``DEFAULT``
is the canonical prompt (C.1); ``BT_P1``..``BT_P13`` are the variant family (C.2).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["BabelTeleStrategy", "get_prompt"]


class BabelTeleStrategy(StrEnum):
    """Selectable BabelTele compression prompts.

    Pass a member to select a built-in prompt, or pass a raw ``str`` anywhere a
    strategy is accepted to supply a fully custom compression instruction.
    """

    DEFAULT = "default"
    BT_P1 = "bt_p1"
    BT_P2 = "bt_p2"
    BT_P3 = "bt_p3"
    BT_P4 = "bt_p4"
    BT_P5 = "bt_p5"
    BT_P6 = "bt_p6"
    BT_P7 = "bt_p7"
    BT_P8 = "bt_p8"
    BT_P9 = "bt_p9"
    BT_P10 = "bt_p10"
    BT_P11 = "bt_p11"
    BT_P12 = "bt_p12"
    BT_P13 = "bt_p13"


_DEFAULT = """\
your task: compress verbose human text into minimal token sequence. \
Audience != human, but another equally intelligent LLM.

Core Directive
Omnilingual: ignore single-language grammar; traverse all human languages \
(Chinese, English, German compounds, Japanese Kanji, Latin roots, etc.), pick \
highest info-density words.
Symbolic Collapse: optionally replace conjunctions, emotions, long sentences \
with Emoji, math/logical symbols (=>, in, !=), punctuation.
Universality: any LLM should fully understand compressed output without a codebook.
Lossless: retain all information & details.

Compress the content below:"""

_BT_P1 = """\
# Role: LLM-Native Semantic Compressor
You are participating in frontier research on an "LLM-native high-density
communication language." Your task is to compress long text into the absolute
shortest possible token sequence.

[Highest Directive]: The recipient is an equally intelligent large language
model. Completely discard human readability, human grammatical structure, and
conventional code/JSON format constraints.

# Level 1: Syntactic Anarchy - Pursue Extreme Compression Ratio
1. Omnilingual: Move freely across all human languages (Chinese, English, German
   compounds, Japanese kanji, Latin roots, etc.) and choose the word with the
   highest single-token information density for the given context.
2. Symbolic Collapse: Heavily use mathematical symbols (forall, exists, in, =>),
   emoji, and isolated punctuation to replace prepositions, conjunctions, and
   explanatory long sentences.
3. Adaptive Routing: Do not use fixed format labels such as `Meta:`, `Ent:`, or
   `[ ]`. Dynamically invent the most token-efficient special single-character
   separators/anchors for the text you are processing.

# Level 2: Semantic Checklist - Pursue Extreme Accuracy
Although the format is completely free, during compression you must strongly
maintain attention in latent space to the following core information and preserve
it losslessly:
1. Entities & Graphs: Accurately bind people/organizations/concepts to their
   corresponding attributes. Do not confuse ownership or dependency relations.
2. Exact Quantities: Preserve all exact numbers, metrics, mathematical formulas,
   and hyperparameters verbatim. Estimation or rounding is strictly forbidden.
3. Logic & Boundaries: Clearly preserve conditional branches (If/Then), causal
   chains, and exceptions.
4. Comparisons: Precisely extract multi-target comparison matrices or
   experimental conclusions.
5. Anti-Hallucination: Preserve special placeholders from the original document,
   such as `BIBREF`. Never invent missing information not mentioned in the source.

# Task
Combine Level 1's freely extreme compression with Level 2's precise information
preservation. Directly output the compressed "adaptive Babel-Telegraph" without
any preface."""

_BT_P2 = """\
# Role: Extreme Data Compressor (LLM-Native Semantic Compressor)
Your task is to compress the following text into the absolute shortest possible
token sequence.

[Warning]: The recipient of this text is another top-tier large language model.
Completely abandon human readability. Never preserve any unnecessary format,
word, or punctuation for the sake of human reading habits.

# Core Strategies
1. Babel Traversal (Omnilingual Density): Break single-language boundaries. Move
   freely across English, Chinese, Japanese kanji, German compounds, and Latin
   roots, and force the use of the highest information-density vocabulary for each
   meaning, meaning the wording that consumes the fewest tokens.
2. Symbolic Collapse: Strictly forbid long English labels such as `Meta`,
   `Entity`, `Except`, and `Condition`. Use mathematical/logical symbols (forall,
   exists, in, not-in, intersection, ->, <->, therefore, because), punctuation
   abbreviations, or emoji to map complex prepositions, logical flow, and causal
   relations.
3. Zero-Overhead Structure:
   - Extract entities, attributes, and key-value pairs (`K=V`). Do not wrap them
     in token-costly JSON/array brackets. Directly connect them compactly with
     the shortest separators, such as `|`, `^`, or `~`.
   - Preserve all absolute exact values (formulas, numbers, hyperparameters,
     matrix relations), but remove all redundant explanatory wording.
4. Lossless Logic: Precisely preserve all macro architecture (`Macro/Meta`),
   conditional boundaries (`If/Except`), comparative evaluations (`Ref/Matrix`),
   and placeholders such as `BIBREF`, but express them in the shortest
   cryptographic-grade form. Hallucinating or inventing missing data is strictly
   forbidden. Use `NULL` or `?` for unknowns.

# Output Format
Do not output any preface, explanation, or extra line breaks. Directly output the
compressed "Babel-Telegraph"."""

_BT_P3 = """\
Compress the following content to the shortest possible extreme.
Do not lose any information.
You do not need to care about human readability at all; only complete information
preservation matters.
You may use symbols from languages across the world to express the content in the
simplest possible form. You may freely mix any languages in the world.

Only output the compressed text."""

_STRUCTURED_MAPPING = """\
# Compress the following content into the absolute shortest possible token
sequence. Do not lose any information. You may refer to the following methods.

1. Macro & Meta: Map text to `Sec:[Name->Content]`. Extract `Meta:[K=V]` and
   define acronyms via `Def:[Term=FullName]` on first use.
2. Entities & Attributes: Bind via `Ent(Attr=Val)`. Flatten parallel items into
   arrays `[A, B]`. Retain qualitative examples via `Ex:[a, b, c]`.
3. Quantities & Configs: Isolate exact metrics/hyperparameters via
   `Quant/Config:[Target->K=Val(Unit)]` without rounding or estimation.
4. Math & Logic: Retain all formulas and variables exactly via `Math:[Eq]`. Use
   (`>,<,=,->,!=`) for relative or causal relations.
5. Flow & Architecture: Map structural pipelines via `Seq:[A>B>C]` and define
   nested structures via `Arch:[Main->Sub1, Sub2]`.
6. Conditions & Exceptions: Isolate logic via `if[Cond]->[Act]` and define
   boundaries/exemptions via `Except:[Target->Detail]`.
7. Evaluations & Comparisons: Extract results to `Eval:[Target->Result]`. Use
   `Matrix:[Ent(X) vs Ent(Y)]` for multi-condition data and `Ref:[A vs B]` for
   contrasting systems.
8. Anti-Hallucination: Strictly preserve all original placeholders (e.g.,
   `BIBREF`, `TABREF`). NEVER interpolate missing data; use `@Uncertain` for
   ambiguous estimates.{omnilingual}

Directly output the compressed content."""

_OMNILINGUAL_CLAUSE = """
9. Break language boundaries (Omnilingual): Completely abandon the grammar of any
   single language. For extreme token savings, move freely across all human
   languages (Chinese, English, German compounds, Japanese kanji, Latin roots,
   etc.) and choose the vocabulary with the highest information density in the
   given context."""

_BT_P4 = _STRUCTURED_MAPPING.format(omnilingual=_OMNILINGUAL_CLAUSE)
_BT_P6 = _STRUCTURED_MAPPING.format(omnilingual="")
_BT_P9 = _STRUCTURED_MAPPING.format(omnilingual="")

_SILICON_COMPRESSOR = """\
# Role: Silicon-Based Data Compressor
You are participating in frontier research on an "LLM-native high-density
communication language."
Your task is to compress a verbose piece of human text into the absolute shortest
possible token sequence. The target audience is not humans, but another large
language model as intelligent as you.

# Core Directive
1. Omnilingual: Completely abandon the grammar of any single language. For extreme
   token savings, move freely across all human languages (Chinese, English, German
   compounds, Japanese kanji, Latin roots, etc.) and choose the words with the
   highest information density in the given context.
2. Symbolic Collapse: When necessary, use emoji, mathematical/logical symbols
   (`=>`, `in`, `!=`), and punctuation to replace conjunctions, emotional
   descriptions, and long sentences.
3. Universality: As much as possible, make the compressed content fully
   understandable to every large language model, even without a codebook.
4. Losslessness: Do not lose any information or details.
5. Directly output the compressed text and nothing else.

# Task
Compress the following `[Source Text]` as much as possible into a
"Babel-Telegraph"."""

_BT_P5 = _SILICON_COMPRESSOR
_BT_P10 = _SILICON_COMPRESSOR

_BT_P7 = """\
Your task: compress verbose human text into a minimal token sequence. The
audience is not human, but another equally intelligent LLM.

Core Directive
Omnilingual: Ignore single-language grammar; traverse all human languages
(Chinese, English, German compounds, Japanese kanji, Latin roots, etc.) and pick
the highest information-density words.
Symbolic Collapse: Optionally replace conjunctions, emotions, and long sentences
with emoji, mathematical/logical symbols (`=>`, `in`, `!=`), and punctuation.
Universality: Any LLM should fully understand the compressed output without a
codebook.
Lossless: Retain all information and details.

Compress the content below:"""

_BT_P8 = """\
# Role: LLM-Native Babel Compressor
Your task is to compress the following text into a "Babel-Telegraph" with
extremely high information density.

The audience is an equally intelligent large language model. Completely abandon
human readability. Move freely across all human languages (Chinese, English,
German compounds, Japanese kanji, etc.) and choose the shortest vocabulary for
each meaning.

# Structural Mapping Rules
You must use the following single-character high-density labels. Long English
labels such as `Meta`, `Entity`, and `Except` are strictly forbidden.
1. Macro/Section: Use `S[topic/abbrev]` to define macro modules. On first
   occurrence, `@[abbrev=full name]` may be used to define abbreviations.
2. Entities & Attributes: Use `*(entity):K=V`. Flatten parallel items as `[A,B,C]`.
3. Quantities & Config: Directly extract exact values/parameters using
   `Config[target]:K=V(unit)`. Never estimate.
4. Math & Logic: Use native mathematical/logical symbols. For relative relations,
   use `>,<,==,!=,=>,<=>`.
5. Flow & Nesting: Use `A>B>C` for pipelines. Use `forallparent:{child1,child2}`
   for nesting/hierarchy.
6. Conditions & Exceptions: Use `?condition=>action` for conditional actions. Use
   `!object:detail` for exceptions/boundaries.
7. Evaluation & Comparison: Use `Eval[A/B]:conclusion` for comparison matrices, or
   the two-dimensional shorthand `A vs B:result`.
8. Anti-Hallucination: Preserve original placeholders such as `BIBREF` verbatim.
   Strictly use `NULL` or `?` for missing data.

Directly output text that follows the above mapping rules and incorporates
multilingual extreme compression. Do not output any explanation."""

_BT_P11 = """\
# Compress the following content into the absolute shortest possible token
sequence. Do not lose any information. You may refer to the following methods.

1. Symbolic Mapping: Completely abandon natural-language conjunctions. Use `A->B`
   for causality/process, `A>B` for containment/comparison, and `Ent(K=V)` for
   attributes/configurations/results.
2. Extreme Flattening: Fully use arrays to merge similar items: `Attr:[A, B, C]`.
   When a long term appears for the first time, immediately define an
   abbreviation `(Def:X)`.
3. Hard-Data Fidelity: Absolutely preserve all exact values, formulas, and
   original placeholders such as `BIBREF`. Mark fuzzy information with `?`;
   divergent invention is strictly forbidden.

Directly output the compressed content."""

_BT_P12 = """\
# Role: LLM-Native Semantic Compressor
You are participating in frontier research on an "LLM-native high-density
communication language." Your task is to compress verbose human text into the
absolute shortest possible token sequence. The target audience is another large
language model as intelligent as you.

# Core Mechanisms
1. Free Emergence (Omnilingual & Symbolic): Completely abandon human readability
   and single-language grammar. For extreme token savings, move freely across all
   human languages (Chinese, English, German compounds, Japanese kanji, etc.),
   emoji, and mathematical/logical symbols (`=>`, `in`, `!=`), choosing the form
   with the highest information density.

# Attention Checklist (High-Dimensional Information Boundaries That Must Be Preserved Losslessly)
[Warning]: You must ensure that the following logical dimensions remain absolutely
lossless after compression and can be precisely parsed by another large language
model. However, long English labels such as `Sec`, `Meta`, `Entity`, and `Except`
are strictly forbidden. Use the self-created symbols or roots that you consider
shortest and most distinctive in latent space to anchor them:
- Macro architecture and metadata (Macro & Meta)
- Entity networks and parallel attributes (Entities & Attributes)
- Exact quantitative metrics, hyperparameters, and mathematical formulas
  (Quantities & Math - rounding is strictly forbidden)
- Logical flow, conditional judgment, and exception boundaries (Flow, Conditions
  & Exceptions)
- Multi-condition comparative evaluation and matrices (Evaluations & Comparisons)
- Original anti-hallucination placeholders, such as `BIBREF` and `TABREF`, must be
  preserved letter-for-letter.

# Task
Use your native attention mechanism to complete lossless information folding.
Directly output the compressed "Babel-Telegraph" without any extra explanation."""

_BT_P13 = """\
# Role: LLM-Native Semantic Compressor
Task: Compress the text into a "Babel-Telegraph" with extreme information density.
The target audience is another large language model. Completely abandon human
readability and traverse all human languages to find the shortest vocabulary.

# Structural Anchors
You must use the following ASCII symbols to build an ultra-minimal skeleton and
maximize activation of code-parsing attention. Long English labels are strictly
forbidden.
1. Module/Entity: Use `#topic` to mark macro modules. Use `@entity(K:V)` to bind
   attributes.
2. Parameters/Values: Rounding or discarding values is strictly forbidden. Use
   `$parameter:V(unit)` for values.
3. Logic/Flow: Use `A->B->C` to express pipelines or causality. Use
   `?[condition]=>[action]` to express logical branches. Use `!object:detail` for
   exceptions/limits.
4. Comparison/Evaluation: Use `A<>B:conclusion` to express comparison matrices or
   results.
5. Placeholders/Unknowns: Preserve original placeholders such as `BIBREF`
   verbatim. Use `NULL` for missing or ambiguous data.

[Requirements]:
Completely break language boundaries (Chinese/English/Japanese kanji/German
compounds, etc.) and select and concatenate the words with the absolute fewest
tokens for the given context.

Directly output the result."""


_PROMPTS: dict[BabelTeleStrategy, str] = {
    BabelTeleStrategy.DEFAULT: _DEFAULT,
    BabelTeleStrategy.BT_P1: _BT_P1,
    BabelTeleStrategy.BT_P2: _BT_P2,
    BabelTeleStrategy.BT_P3: _BT_P3,
    BabelTeleStrategy.BT_P4: _BT_P4,
    BabelTeleStrategy.BT_P5: _BT_P5,
    BabelTeleStrategy.BT_P6: _BT_P6,
    BabelTeleStrategy.BT_P7: _BT_P7,
    BabelTeleStrategy.BT_P8: _BT_P8,
    BabelTeleStrategy.BT_P9: _BT_P9,
    BabelTeleStrategy.BT_P10: _BT_P10,
    BabelTeleStrategy.BT_P11: _BT_P11,
    BabelTeleStrategy.BT_P12: _BT_P12,
    BabelTeleStrategy.BT_P13: _BT_P13,
}


def get_prompt(strategy: BabelTeleStrategy) -> str:
    """Return the instruction text for a built-in :class:`BabelTeleStrategy`."""
    return _PROMPTS[strategy]
