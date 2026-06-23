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
