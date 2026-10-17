## decisions-reranker

A simple reranker for RAG systems, built on top of the OpenAI Decisions API.

OpenAI doesn't offer a dedicated reranking model, so this package uses the Decisions API to fill that gap. If your stack already runs on OpenAI, you can add reranking without bringing in another provider.

### requirements

This package calls the OpenAI API and requires an OpenAI API key. By default it creates an `OpenAI()` client, which reads the key from the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

Alternatively, pass your own configured client:

```python
from openai import OpenAI
from decisions_reranker import DecisionsReranker

reranker = DecisionsReranker(client=OpenAI(api_key="sk-..."))
```

Each `filter()` or `rerank()` call makes one request to the Decisions API and is billed to your OpenAI account.

### methods

```python
DecisionsReranker().filter()
```

Filters a list of documents for relevance regarding a given query. Returns only documents above a given threshold.

```python
DecisionsReranker().rerank()
```

Reorders a list of documents by relevance regarding a given query. Returns all documents in a new order.

### parameters

The DecisionsReranker constructor takes the following parameters. All of them are optional:

```python
    client: OpenAI | None = None,
    model: str = "gpt-6-luna",
    instruction: str = DEFAULT_INSTRUCTION,
    threshold: float = 0.5
```

### Example usage

```python
from decisions_reranker import DecisionsReranker, Document

reranker = DecisionsReranker(
    instruction="Is this snippet relevant to answer the given query: {document}. Only answer with true if the snippet really contains directly relevant information for the question",
    threshold=0.8,
)

query = "How can I reduce memory usage in a Python application?"

documents = [
    { "id": "d01", "text": "Python generators yield items lazily, so you avoid loading an entire list into memory at once."},
    { "id": "d02", "text": "Defining __slots__ on a Python class removes the per-instance __dict__ and cuts memory overhead."},
    { "id": "d03", "text": "The tracemalloc module shows which lines of Python code allocate the most memory."},
    { "id": "d04", "text": "Python's GIL prevents multiple threads from executing bytecode in parallel."},
    { "id": "d05", "text": "NumPy arrays store numbers far more compactly than Python lists of int objects."}
]

documents = [Document(id=d["id"], text=d["text"]) for d in documents]

filtered =  reranker.filter(query, documents)

reranked = reranker.rerank(query, documents)

```