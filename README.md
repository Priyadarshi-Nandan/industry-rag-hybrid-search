# Local RAG System

A local-first Retrieval-Augmented Generation (RAG) pipeline that ingests documents (TXT, PDF, DOCX), retrieves relevant evidence using hybrid search, and generates grounded, cited answers using a local LLM — no cloud APIs, no data leaving your machine.

## Architecture

```
Ingestion:
  file (.txt / .pdf / .docx)
    -> text extraction (pypdf / python-docx / plain read)
    -> heading-aware recursive chunking
    -> embedding (Ollama: nomic-embed-text)
    -> stored in ChromaDB (persistent vector store)

Retrieval:
  question
    -> embedded (nomic-embed-text)
    -> vector search (ChromaDB)          -----+
    -> keyword search (BM25)             -----+--> merged candidates
    -> reranked (cross-encoder, ms-marco-MiniLM-L-6-v2)
    -> top chunks -> prompt -> local LLM (qwen2.5:3b via Ollama)
    -> grounded answer with source citation
```

## Stack and links

| Component | Tool | Link |
|---|---|---|
| Local LLM runtime | Ollama | https://ollama.com |
| Generation model | qwen2.5:3b | https://ollama.com/library/qwen2.5 |
| Embedding model | nomic-embed-text | https://ollama.com/library/nomic-embed-text |
| Vector database | ChromaDB | https://www.trychroma.com |
| Keyword search | rank_bm25 | https://pypi.org/project/rank-bm25 |
| Reranker | sentence-transformers (cross-encoder) | https://www.sbert.net |
| PDF parsing | pypdf | https://pypi.org/project/pypdf |
| DOCX parsing | python-docx | https://python-docx.readthedocs.io |

## Setup

If you want to run this on your own computer, follow these steps:

1. Clone this repository
2. Install Ollama: https://ollama.com/download
3. Pull models:
   ```
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```
4. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
5. Install dependencies:
   ```
   pip install ollama chromadb rank_bm25 sentence-transformers pypdf python-docx python-dotenv
   ```
6. Create a `.env` file:
   ```
   DOCS_FOLDER=docs
   EMBED_MODEL=nomic-embed-text
   GEN_MODEL=qwen2.5:3b
   ```
7. Add documents to the `docs` folder.
8. Run:
   ```
   python rag.py
   ```

## Known limitations

- Small local models occasionally abstain even when evidence is present (model inconsistency, not a retrieval failure — verified by rerunning identical queries).
- Terminal-only interface, no web UI yet.
- Not tested against very large document sets or non-text PDF content (scanned pages, charts).

## Future plans

- Query rewriting
- Multi-hop retrieval
- Conversation memory
- Token usage tracking
- Metadata-based source citations
- Hosted frontend with document upload

---

## My journey building this

### The very first version

Started with the simplest possible version: Python, Ollama, a 1B model, and three short text files about towers. No framework, no database — just NumPy computing cosine similarity by hand between embeddings. It worked, and more importantly, it correctly refused to answer a question that wasn't covered by the documents ("I don't have enough information"), which is really the whole point of RAG — I wanted to see that abstention behavior actually happen before trusting the rest of the pipeline.

### Making it more "real"

From there I rebuilt it into something closer to how an actual system would be structured. Swapped the manual NumPy search for ChromaDB, added BM25 so keyword matches aren't missed when they don't line up semantically, and added a cross-encoder reranker to re-score the top candidates before they go to the model. Moved from a 1B model to qwen2.5:3b, since the bigger model followed the "answer only from context" instruction more reliably. Added PDF and DOCX support, and a hash-based check so re-running the script doesn't re-embed documents that haven't changed.

### The chunking bug (the annoying one)

This was the most useful bug I ran into. My sentence-based chunker didn't know anything about document structure, so the last sentence of one chapter sometimes got merged into the same chunk as the heading and first sentence of the next chapter. In my case, the end of a section about Mars got fused with the start of a section about Jupiter. The embedding for that chunk ended up representing a mix of both topics, so it didn't score well for either — the model couldn't reliably answer "which planet is the largest" because the correct chunk was sitting there, just diluted.

Fixed it with recursive, heading-aware chunking, which works like a fallback ladder — it only drops to a smaller unit when the current one is too big:

```
document text
  -> split by heading (Chapter 1, Chapter 2, ...)
       is section small enough?
         yes -> keep whole section as one chunk
         no  -> split section by paragraph
                  is paragraph small enough?
                    yes -> group paragraphs together up to chunk size
                    no  -> split paragraph by sentence
                             group sentences together up to chunk size
```

Because splitting always starts at the heading level and only goes smaller when needed, a chunk can never contain text from two different chapters at once. That's what directly fixed the Mars/Jupiter mixing — the old sentence-only chunker had no concept of "chapter," so it just counted sentences and cut wherever the count ran out, regardless of what topic it crossed into.

Re-ran the same failing question afterward and got the correct answer, cited to the right chapter. Genuinely satisfying to catch and fix.

### Where it's at now

Current state is a working local pipeline covering ingestion, hybrid retrieval, reranking, and grounded generation. Next up is cleaning up the terminal output, adding source metadata to answers, and eventually building this into a proper hosted app.
