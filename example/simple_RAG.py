import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai



load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_FILE = PROJECT_DIR / "data" / "sample_dokumen.txt"


# =========================
# 1. LOAD DATA
# =========================
def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Expected location: {DATA_FILE}"
        )

    with path.open("r", encoding="utf-8") as f:
        texts = f.readlines()
    return [t.strip() for t in texts if t.strip()]



# =========================
# 2. PREPROCESS (TODO)
# =========================
def preprocess(texts):
    # TODO: mahasiswa bisa tambahkan cleaning
    return texts


# =========================
# 3. CHUNKING (TODO)
# =========================
def chunking(texts, chunk_size=2):
    chunks = []
    for i in range(0, len(texts), chunk_size):
        chunk = " ".join(texts[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# =========================
# 4. EMBEDDING
# =========================
def create_embeddings(chunks, model):
    return model.encode(chunks)


# =========================
# 5. VECTOR STORE (FAISS)
# =========================
def build_index(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index



# =========================
# 6. RETRIEVAL
# =========================
def retrieve(query, model, index, chunks, k=2):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    results = [chunks[i] for i in indices[0]]
    return results


# =========================
# 7. SIMPLE QA (NO LLM)
# =========================
def answer_question(query, context):
    # Versi sederhana (tanpa OpenAI)
    return f"Pertanyaan: {query}\nJawaban berdasarkan konteks:\n{context}"


# -----------------------------
# 7. LLM ANSWER
# -----------------------------
def answer_with_llm(query, context_chunks):
    context_text = "\n".join(context_chunks)
    prompt = f"Jawab pertanyaan berikut berdasarkan konteks:\n{context_text}\nPertanyaan: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# =========================
# MAIN PIPELINE
# =========================
def main():
    # Load model embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 1: Load data
    texts = load_data(DATA_FILE)

    # Step 2: Preprocess
    texts = preprocess(texts)

    # Step 3: Chunking
    chunks = chunking(texts)

    # Step 4: Embedding
    embeddings = create_embeddings(chunks, model)

    # Step 5: Indexing
    index = build_index(embeddings)

    # Step 6: Query loop
    while True:
        query = input("\nTanya (ketik 'exit' untuk keluar): ")
        if query.lower() == "exit":
            break

        context = retrieve(query, model, index, chunks)
        answer = answer_question(query, context)

        print("\n=== HASIL ===")
        print(answer)


    # print("=== Sistem RAG + LLM siap digunakan ===")
    #     while True:
    #         query = input("\nTanya (ketik 'exit' untuk keluar): ")
    #         if query.lower() == "exit":
    #             break
    #         context = retrieve(query, model, index, chunks)
    #         answer = answer_with_llm(query, context)
    #         print("\n=== HASIL ===")
    #         print(answer)

if __name__ == "__main__":
    main()