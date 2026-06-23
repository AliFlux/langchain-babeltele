# Examples

Runnable scripts showing where BabelTele plugs in.

## Setup

```bash
pip install langchain-babeltele python-dotenv
pip install langchain-anthropic   # or your provider's package
```

Edit `default.env` and add your API key, or copy it to `.env` (which overrides
`default.env` and is git-ignored):

```bash
cp default.env .env
```

## Scripts

Start with `hello.py`, then work down.

- [hello.py](hello.py) — the smallest example: compress one paragraph and print the
  before/after with token counts.

```bash
python examples/hello.py
```

- [strategies.py](strategies.py) — run the same fact-dense text through several
  `BT_P*` prompt strategies and print a side-by-side density comparison.

```bash
python examples/strategies.py
```

- [guardrail.py](guardrail.py) — wrap compression in a `FidelityGuardrail` that
  scores recoverability with an LLM judge, retries with milder strategies, and falls
  back to the original text if it can't ensure fidelity. Prints the `verified` flag.

```bash
python examples/guardrail.py
```

- [compress_history.py](compress_history.py) — compress a multi-turn agent
  conversation into a dense BabelTele digest, printing the before/after text and
  token savings. This is the operation `BabelTeleCompressionMiddleware` performs
  automatically inside `create_agent`.

```bash
python examples/compress_history.py
```

- [long_term_memory.py](long_term_memory.py) — store several past sessions as
  compressed, embedded memories, then retrieve only the relevant one for a later
  question. Prints each session's compression and the dense text that grounds the
  answer. Needs both a chat model (`BABELTELE_MODEL`) and an embedding model
  (`BABELTELE_EMBEDDINGS`), so install your embedding provider too:

```bash
pip install langchain-openai   # for the default openai embeddings
python examples/long_term_memory.py
```
