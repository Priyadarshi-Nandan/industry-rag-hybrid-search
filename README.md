# Local RAG System

A local-first Retrieval-Augmented Generation (RAG) pipeline that ingests documents (TXT, PDF, DOCX), retrieves relevant evidence using hybrid search, and generates grounded, cited answers using a local LLM — no cloud APIs, no data leaving your machine.

## Features

- Multi-format ingestion: TXT, PDF, DOCX
- Heading-aware chunking, every chunk carries its source file and section heading
- Hybrid retrieval: vector search plus BM25 keyword search
- Cross-encoder reranking of candidate chunks
- Smart re-indexing: only new or changed files get embedded again
- Multi-question chat loop in the terminal
- Conversation memory: follow-up questions like "how big is it?" work
- Query rewriting: follow-ups are turned into standalone questions before searching
- Source citations with file and chapter, hidden when the model abstains
- Low-confidence sources are labeled instead of pretending to be solid evidence
- Token usage tracking per question and per session
- A fixed 15-question test set that scores the whole pipeline and saves a report with timing and token counts

## Example session

```
Ask a question: tell me about jupiter
[Searching for: tell me about jupiter]

Answer: Jupiter is the fifth planet from the Sun and the largest in the Solar System.
It is a gas giant with a mass more than two and a half times that of all the other
planets in the Solar System combined...

Sources:
  - solar_system.pdf → Chapter 7: Jupiter

[Tokens] prompt: 798 | answer: 159 | this question: 957 | session total: 957 | time: 4.3s

Ask a question: how many moons does it have
[Searching for: How many moons does Jupiter have?]

Answer: Jupiter has 95 known moons, according to the information provided in
solar_system.pdf under Chapter 15: Moons of the Solar System.

Sources:
  - solar_system.pdf → Chapter 15: Moons of the Solar System, Chapter 7: Jupiter

[Tokens] prompt: 868 | answer: 40 | this question: 908 | session total: 1865 | time: 0.9s
```

The `[Searching for: ...]` line shows what the follow-up question was rewritten into before the search ran.

## Architecture

```
Ingestion:
  file (.txt / .pdf / .docx)
    -> text extraction (pypdf / python-docx / plain read)
    -> heading-aware recursive chunking
    -> embedding (Ollama: nomic-embed-text)
    -> stored in ChromaDB (persistent vector store)

Retrieval:
  question + recent conversation history
    -> query rewriting (local LLM turns "how big is it?" into "How big is Jupiter?")
    -> embedded (nomic-embed-text)
    -> vector search (ChromaDB)          -----+
    -> keyword search (BM25)             -----+--> merged candidates
    -> reranked (cross-encoder, ms-marco-MiniLM-L-6-v2)
    -> top chunks + last 3 exchanges -> prompt -> local LLM (qwen2.5:3b via Ollama)
    -> grounded answer
    -> sources (file + heading, only if the model actually answered)
    -> answer saved to conversation history
    -> token usage report
    -> back to the next question
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

## Testing

`test_set.py` runs 15 fixed questions through the same pipeline as the chat loop and saves a report file with timing, token counts, and full details for any question that fails. The questions are 9 normal ones across the documents, 2 that the documents can't answer (the system should refuse), and 2 short follow-up conversations (4 questions) to test memory and query rewriting.

Latest run (qwen2.5:3b, 3 chunks sent to the model per question):

| Measure | Result |
|---|---|
| Score | 14 / 15 |
| Right chunk reached the model | 13 / 13 |
| Correct refusals | 2 / 2 |
| Average time per question | 1.2 s |
| Average tokens per question | 447 |

The one failure (the length of the Great Wall) had the right chunk ranked first by a wide margin, and the 3B model still said it didn't have enough information. So that miss is on the model side, not the search side.

Sending only the top 3 chunks to the model, instead of every candidate, cut tokens per question from about 809 to 447. The score stayed at 14 / 15, though one question started passing and another stopped. I wrote the questions myself for these documents, so this is a regression check for my own changes, not a general accuracy number.

## Known limitations

- Small local models occasionally abstain even when evidence is present (model inconsistency, not a retrieval failure — verified by rerunning identical queries).
- Very short or vague questions (like a single word) can retrieve weak chunks and end in an abstention, even when the answer exists.
- Query rewriting works well on clean follow-ups, but the 3B model can miss on garbled sentences or heavy typos (for example, leaving "this planet" unresolved). A larger model should help.
- Questions that need counting or combining facts across chapters (like "how many gas giants are there?") often end in an abstention, since no single chunk states the answer.
- Memory keeps only the last 3 exchanges, to keep the prompt size under control.
- Terminal-only interface, no web UI yet.
- Not tested against very large document sets or non-text PDF content (scanned pages, charts).

## Future plans

- Skip the LLM call when retrieval confidence is very low (saves tokens)
- Split the code into separate ingestion and retrieval modules
- Multi-hop retrieval
- Rerun the test set on a larger local model (6-8B) and compare against the numbers above
- Hosted frontend with document upload
- Rebuild a smaller version with LangChain and LangGraph, to compare against the from-scratch version

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

### Getting the sources right (the other annoying one)

After the chunking fix I added source citations, and they kept showing the wrong thing. Asking about the Sun listed "general content" instead of "Chapter 2: The Sun," even though the database clearly stored the right heading for every chunk. I confirmed that by querying ChromaDB directly, which ruled out the chunker and the database.

The actual bug was in the hybrid search merge. The BM25 results were built with only the text and the source file, no heading, and when the same chunk came back from both searches, the BM25 version overwrote the vector version. So any chunk BM25 found silently lost its heading. One missing key in one dictionary. Adding the heading to the BM25 results fixed it, and the lesson was to check each stage separately before touching the code.

### What the token counter showed

I added a token counter to see where the cost goes. The prompt is usually 3 to 7 times bigger than the answer, because the retrieved chunks fill it. A question the model can't answer still spent about 590 tokens just to say "I don't have enough information." That gives me a clear first optimization target: skip the model call when the retrieved chunks are clearly not relevant.

### Adding memory, and finding out it wasn't enough

Once the loop worked, I wanted follow-up questions to work, like asking about Jupiter and then "what is its size?". The first step was easy: keep the last few question and answer pairs and put them in the prompt. That helped the model understand what "it" meant, but follow-ups still failed. The reason is that the search never sees the conversation. It looked up "size of this planet" with no mention of Jupiter, so the right chunk never reached the model, and the model correctly said it didn't have enough information.

So memory has two halves: the model needs to understand the follow-up, and the search needs to understand it too.

### Query rewriting

The fix was to rewrite the follow-up into a standalone question before searching, using the local model itself: "what is its size" becomes "What is the size of Jupiter?". The first version rewrote inconsistently. The 3B model would sometimes leave "it" in place, or turn "this planet" into "this planet" again. Adding worked examples to the rewrite prompt fixed most of it, since small models follow examples much better than rules. One example is a question that should come back unchanged, because without it a standalone question about Mount Everest got rewritten to borrow words from the previous topic.

I also tried gating the rewrite on a list of pronouns to save model calls, and dropped it. My own typing has plenty of typos ("thsi planet"), and a typo slipped straight past the pronoun list. The rewriter turned out to be good at fixing typos, so letting it run on every follow-up worked better than trying to be clever.

### The model that was too cautious, then too chatty

Tuning the prompt was a balancing act. With strict wording, the model said "I don't have enough information" even when the answer was right there, because the document says "diameter" and I asked about "size". Loosening the instruction fixed that, but then it started saying "I don't have enough information" followed by the answer anyway. The last change was to tell it to either answer or abstain, never both, and to only count an answer as an abstention if it starts with that phrase.

One more thing I changed along the way: when nothing scores well, the system used to list the top two chunks as sources anyway. Those looked like real evidence even when the answer came from the earlier conversation. Now those are labeled low confidence instead.

### Testing it properly

Up to this point I'd been judging every change by typing a few questions and reading the answers, which doesn't tell you much. So I wrote a fixed set of 15 questions with an expected keyword for each, and a script that runs them and writes a report. It checks two things separately, whether the right chunk reached the model and whether the final answer was right, so a failure points at either the search or the model.

It paid off straight away. A follow-up like "which animals live there?" wasn't being rewritten, and the test caught it. Sending fewer chunks to the model cut the prompt size by almost half, and the test showed the score held. It also showed that the one remaining failure is the model refusing with the right text in front of it, which is useful to know before spending time tuning search.

### Where it's at now

Current state is a working local pipeline covering ingestion, hybrid retrieval, reranking, grounded generation, source citations with headings, a multi-question loop with conversation memory, query rewriting, and token tracking. On clean follow-ups it works reliably. On messy ones, the 3B model is the limit, and trying a larger model is on the list. Next up is testing a larger model against the same questions, and then a simple hosted version.
