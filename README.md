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

## Examples

If you learn faster from running code, start in [examples/](examples/). The
[hello.py](examples/hello.py) script is the smallest end-to-end run: compress one
paragraph and print the before/after with token counts.

```bash
cp examples/default.env examples/.env   # add your API key
python examples/hello.py
```

From there, [examples/](examples/) builds up through prompt strategies, the fidelity
guardrail, agent-history compression, and long-term memory. See the
[examples README](examples/README.md) for the full list.

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

## Benchmarks

**On document QA, BabelTele kept 99.5% semantic fidelity while
compressing text to 27.9% of its original length.**

**Agent memory (LoCoMo).** Compressing each session before storing it retains most
of the full-text accuracy at roughly half the tokens, and edges out plain
summarization. This is what `BabelTeleMemoryStore` does.

Results depend on the compressor and reader models and on the task, so treat these
as indicative rather than guarantees.

| Method    | Tokens / query | Accuracy | vs. original |
| --------- | -------------: | -------: | -----------: |
| Original  |         2819.5 |    64.81 |       100.0% |
| Summary   |         1365.6 |    61.05 |        94.2% |
| BabelTele |         1382.2 |    62.53 |        96.5% |

**Multi-agent communication.** Compressing inter-agent messages cut tokens sharply
with little score loss.

| Setting                          | Token reduction | Score (vs. uncompressed) |
| -------------------------------- | --------------: | -----------------------: |
| Homogeneous (Gemini with Gemini) |          38.96% |                    96.6% |
| Heterogeneous (Gemini with GPT)  |          44.21% |                    99.7% |

**Beyond the context window.** When the input exceeds the window, chunked BabelTele
compression beat naive truncation on LongBench v2 Code Repo QA (Long). This is the
`chunk_tokens` path.

| Reader      | Truncation | BabelTele |
| ----------- | ---------: | --------: |
| Qwen3.6-Max |      55.17 |     62.07 |
| GLM-5.1     |      62.07 |     72.41 |
| Kimi 2.5    |      44.82 |     48.28 |

**Compression strength varies by model.** On LongBench v2, Gemini 3.1 Pro was the
most aggressive at over 95% compression (about 4% retention), while GPT-5.4 was the
most conservative at roughly 75% (about 27% retention); other models landed in
between. Stronger compressors such as GPT and Claude also produced the most portable
output for other readers to decode.

All numbers are from the paper ([arXiv:2606.19857](https://arxiv.org/abs/2606.19857)). Pull requests are welcome to develop evals for this project and generate more results.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
