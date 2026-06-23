# langchain-babeltele

Model-native, high-density text compression for LangChain pipelines and agents.

BabelTele compresses verbose text into a dense form that people can't read but
LLMs can recover. It relaxes the readability prior, using omnilingual word choice
and symbolic collapse, then lets downstream models consume the dense text directly
with no decompression step. The paper reports ~28% of the original token length at
~99.5% downstream QA fidelity, with no fine-tuning and pure black-box API access.

Based on *"Large Language Models Do Not Always Need Readable Language"*
([arXiv:2606.19857](https://arxiv.org/abs/2606.19857)).

## Install

```bash
pip install langchain-babeltele
```

You also need a chat-model provider, e.g. `pip install langchain-anthropic`.

## The core engine

Everything composes from one primitive. Pass any chat model or a model string
(resolved via `init_chat_model`).

```python
from langchain_babeltele import BabelTeleCompressor

compressor = BabelTeleCompressor("anthropic:claude-sonnet-4-6")
result = compressor.compress(long_text)

print(result.text)             # the dense BabelTele representation
print(result.retention_ratio)  # e.g. 0.28

# Use it anywhere in an LCEL chain:
chain = compressor.as_runnable() | some_reader_model
```

Long inputs that exceed the compressor's own context window are chunked
automatically:

```python
BabelTeleCompressor(model, chunk_tokens=200_000)
```

## Where it plugs in

### RAG: compress retrieved documents

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_babeltele import BabelTeleDocumentCompressor

retriever = ContextualCompressionRetriever(
    base_compressor=BabelTeleDocumentCompressor(compressor=compressor),
    base_retriever=base_retriever,
)
```

### Agents: compress history and tool outputs

A denser drop-in alternative to `SummarizationMiddleware`. Folds overflowing
history into one dense message and compresses large tool outputs before they
re-enter context.

```python
from langchain.agents import create_agent
from langchain_babeltele import BabelTeleCompressionMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=tools,
    middleware=[
        BabelTeleCompressionMiddleware(
            compressor,
            token_budget=4000,
            keep_last_n=2,
            tool_output_threshold=2000,
        )
    ],
)
```

### Long-term memory

The paper's LoCoMo recipe: compress each session, embed, retrieve top-k.

```python
from langchain_babeltele import BabelTeleMemoryStore

memory = BabelTeleMemoryStore(vector_store, compressor)
memory.add_session(conversation_text)
relevant = memory.retrieve("what did we decide about pricing?", k=4)
```

## Choosing a prompt strategy

BabelTele offers several prompt strategies rather than one fixed prompt. Select a
built-in strategy or pass your own:

```python
from langchain_babeltele import BabelTeleStrategy

BabelTeleCompressor(model, strategy=BabelTeleStrategy.BT_P8)   # fixed symbolic rules
BabelTeleCompressor(model, strategy="my custom compression prompt: ")
```

## Fidelity guardrail

Because BabelTele abandons readability, a faulty compression can silently drop
information. The guardrail scores recoverability with an LLM judge and retries
with milder structured strategies. If it can't ensure fidelity, it falls back to
the original text.

```python
from langchain_babeltele import FidelityGuardrail

compressor = BabelTeleCompressor(
    model,
    guardrail=FidelityGuardrail("anthropic:claude-sonnet-4-6", threshold=0.8),
)
result = compressor.compress(text)
print(result.verified)  # True / False
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
