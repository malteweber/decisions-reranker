## decisions-reranker

A simple package to use the decisions api from openai as a reranker for RAG systems.

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