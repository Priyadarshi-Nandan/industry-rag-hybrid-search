import os
import re
import ollama
import chromadb
import hashlib
from docx import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

DOCS_FOLDER = os.getenv("DOCS_FOLDER")
EMBED_MODEL = os.getenv("EMBED_MODEL")
GEN_MODEL = os.getenv("GEN_MODEL")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

from docx import Document

def read_docx(path):
    doc = Document(path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def chunk_text(text, chunk_size=500):
    sections = re.split(r'(?=^#{1,3}\s|^Chapter \d+:)', text, flags=re.MULTILINE)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        heading_match = re.match(r'^(#{1,3}\s.+|Chapter \d+:.+)', section)
        if heading_match:
            heading = heading_match.group(1).strip()
            content = section[heading_match.end():].strip()
        else:
            heading = "Introduction / Untitled Section"
            content = section

        if len(section) <= chunk_size:
            chunks.append({"text": section, "heading": heading})
            continue

        paragraphs = content.split("\n\n")
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 2 <= chunk_size:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append({"text": f"[{heading}] {current.strip()}", "heading": heading})
                if len(para) > chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current = ""
                    for sentence in sentences:
                        if len(current) + len(sentence) + 1 <= chunk_size:
                            current += sentence + " "
                        else:
                            if current:
                                chunks.append({"text": f"[{heading}] {current.strip()}", "heading": heading})
                            current = sentence + " "
                else:
                    current = para + "\n\n"
        if current:
            chunks.append({"text": f"[{heading}] {current.strip()}", "heading": heading})

    return chunks

def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_documents():
    docs = []
    for filename in os.listdir(DOCS_FOLDER):
        path = os.path.join(DOCS_FOLDER, filename)
        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".pdf"):
            text = read_pdf(path)
        elif filename.endswith(".docx"):
            text = read_docx(path)
        else:
            continue
        pieces = chunk_text(text)
        for i, piece in enumerate(pieces):
            docs.append({"id": f"{filename}#chunk{i}", "source": filename, "text": piece["text"], "heading": piece["heading"]})
    return docs

def build_bm25_index(docs):
    tokenized = [doc["text"].lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25

def embed_text(text):
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]

def index_documents():
    docs = load_documents()
    print(f"Loaded {len(docs)} chunks")
    for doc in docs:
        file_path = os.path.join(DOCS_FOLDER, doc["source"])
        file_hash = get_file_hash(file_path)
        existing = collection.get(ids=[doc["id"]])
        if existing["ids"] and existing["metadatas"][0].get("hash") == file_hash:
            continue
        embedding = embed_text(doc["text"])
        collection.upsert(
            ids=[doc["id"]],
            embeddings=[embedding],
            documents=[doc["text"]],
            metadatas=[{"source": doc["source"], "hash": file_hash, "heading": doc.get("heading") or "Introduction / Untitled Section"}]
        )
    print("Indexed all new/changed chunks into ChromaDB")

def retrieve(question, top_k=4):
    q_embedding = embed_text(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    matches = []
    for i in range(len(results["documents"][0])):
        matches.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "heading": results["metadatas"][0][i].get("heading", ""),
            "distance": results["distances"][0][i]
        })
    return matches

def hybrid_retrieve(question, docs, bm25, top_k=4):
    # Vector search
    vector_results = retrieve(question, top_k=top_k)

    # BM25 keyword search
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_results = [{"text": docs[i]["text"], "source": docs[i]["source"], "heading": docs[i]["heading"]} for i in top_bm25_indices]

    # Combine, removing duplicates by text
    combined = {r["text"]: r for r in vector_results}
    for r in bm25_results:
        combined[r["text"]] = r

    return list(combined.values())

def rerank(question, matches):
    pairs = [[question, m["text"]] for m in matches]
    scores = reranker.predict(pairs)
    for m, score in zip(matches, scores):
        m["rerank_score"] = score
    matches.sort(key=lambda m: m["rerank_score"], reverse=True)
    return matches

def build_prompt(question, matches):
    context = "\n\n".join([f"[{m['source']}]: {m['text']}" for m in matches])
    prompt = "Answer the question using ONLY the context below.\n"
    prompt += "If the context doesn't contain the answer, say \"I don't have enough information.\"\n"
    prompt += "Cite the source filename in your answer.\n\n"
    prompt += f"Context:\n{context}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Answer:"
    return prompt

if __name__ == "__main__":
    docs = load_documents()
    bm25 = build_bm25_index(docs)
    index_documents()

    print("\nRAG is ready. Type 'exit' to stop.")

    session_tokens = 0

    while True:
        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            break

        matches = hybrid_retrieve(question, docs, bm25)
        matches = rerank(question, matches)

        prompt = build_prompt(question, matches)
        response = ollama.generate(
            model=GEN_MODEL,
            prompt=prompt,
            options={"temperature": 0, "num_predict": 300}
        )

        top_matches = [m for m in matches if m.get("rerank_score", 0) > 0]
        if not top_matches:
            top_matches = matches[:2]

        source_map = {}
        for m in top_matches:
            file = m["source"]
            heading = m.get("heading")
            if heading == "Introduction / Untitled Section":
                heading = None
            source_map.setdefault(file, set())
            if heading:
                source_map[file].add(heading)

        source_lines = []
        for file, headings in sorted(source_map.items()):
            if headings:
                source_lines.append(f"{file} → {', '.join(sorted(headings))}")
            else:
                source_lines.append(file)

        is_abstention = "i don't have enough information" in response["response"].lower()


        print(f"\nAnswer: {response['response']}")

        if not is_abstention:
            print("\nSources:")
            for line in source_lines:
                print(f"  - {line}")


        prompt_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)
        seconds = response.get("total_duration", 0) / 1e9
        session_tokens += prompt_tokens + output_tokens

        print(f"\n[Tokens] prompt: {prompt_tokens} | answer: {output_tokens} | this question: {prompt_tokens + output_tokens} | session total: {session_tokens} | time: {seconds:.1f}s")
