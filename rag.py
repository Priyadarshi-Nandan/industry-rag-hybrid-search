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
            heading = ""
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
            metadatas=[{"source": doc["source"], "hash": file_hash}]
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
    bm25_results = [{"text": docs[i]["text"], "source": docs[i]["source"]} for i in top_bm25_indices]

    # Combine, removing duplicates by text
    combined = {r["text"]: r for r in vector_results}
    for r in bm25_results:
        combined[r["text"]] = r

    return list(combined.values())

def rerank(question, matches):
    pairs = [[question, m["text"]] for m in matches]
    scores = reranker.predict(pairs)
    print("Rerank scores:", scores)
    scored = list(zip(scores, matches))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for score, m in scored]

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

    question = input("\nAsk a question: ")
    matches = hybrid_retrieve(question, docs, bm25)
    matches = rerank(question, matches)

    print(f"\nQuestion: {question}\n")
    for m in matches:
        dist = m.get("distance", "N/A (BM25 match)")
        print(f"Distance: {dist} | Source: {m['source']}")
        print(m["text"])
        print("---")

    prompt = build_prompt(question, matches)
    # response = ollama.generate(model=GEN_MODEL, prompt=prompt)
    response = ollama.generate(
    model=GEN_MODEL,
    prompt=prompt,
    options={"temperature": 0, "num_predict": 300}
)
    print("\n=== ANSWER ===")
    print(response["response"])