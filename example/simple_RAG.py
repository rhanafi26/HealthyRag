import csv
import os
from pathlib import Path

import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_FILE = PROJECT_DIR / "data" / "1739240888.csv"


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

    data = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                data.append(row)

    return data



# =========================
# 2. PREPROCESS
# =========================
def detect_type(value):
    try:
        if "." in value:
            return float(value)
        return int(value)
    except:
        return value

def clean_text(text):
    # lowercase text
    text = text.lower()
    #  hapus tanda baca
    text = re.sub(r"[^\w\s]", "", text)
    # hapus spasi
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_url(text):
    return text.startswith("http")

def preprocess(texts):
    cleaned_texts = []

    # ambil header
    header = texts[0]

    # start index ke 1
    for row in texts[1:]:
        # skip row yang tidak valid
        if len(row) != len(header):
            continue

        item = {}

        for i, col in enumerate(header):
            # get data by index
            value = row[i]

            # deteksi tipe data value
            value = detect_type(value)

            # membersihkan value agar seragam
            if isinstance(value, str) and not is_url(value):
                value = clean_text(value)

            # masukan ke dict agar jadi json
            item[col] = value

        text = " ".join([f"{k} {v}" for k, v in item.items()])

        cleaned_texts.append(text)

    return cleaned_texts


    # return cleaned_texts

# =========================
# 3. CHUNKING
# =========================
def chunking(texts, chunk_size=4):
    chunks = []
    for i in range(0, len(texts), chunk_size):
        # chunk = " ".join(texts[i:i+chunk_size])
        chunk = texts[i:i + chunk_size]
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
def retrieve(query, model, index, chunks, k=4):
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
def flatten_chunks(chunks):
    result = []
    for chunk in chunks:
        if isinstance(chunk, list):
            result.extend(chunk)
        else:
            result.append(chunk)
    return result

def answer_with_llm(query, context_chunks):
    context_text = "\n".join(flatten_chunks(context_chunks))

    prompt = f"Jawab pertanyaan berikut berdasarkan konteks:\n{context_text}\nPertanyaan: {query}"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content

# =========================
# MAIN PIPELINE
# =========================
def main():
    # Load model embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 1: Load data
    texts = load_data(DATA_FILE)
    print(texts)

    # Step 2: Preprocess
    texts = preprocess(texts)
    print(texts)

    # # Step 3: Chunking
    chunks = chunking(texts)
    print(chunks)

    # # Step 4: Embedding
    embeddings = create_embeddings(chunks, model)
    print(embeddings)

    # # Step 5: Indexing
    index = build_index(embeddings)
    print(index)


    #
    # # Step 6: Query loop
    # while True:
    #     query = input("\nTanya (ketik 'exit' untuk keluar): ")
    #     if query.lower() == "exit":
    #         break
    #
    #     context = retrieve(query, model, index, chunks)
    #     answer = answer_question(query, context)
    #
    #     print("\n=== HASIL ===")
    #     print(answer)


    print("=== Sistem RAG + LLM siap digunakan ===")
    while True:
        query = input("\nTanya (ketik 'exit' untuk keluar): ")
        if query.lower() == "exit":
            break
        context = retrieve(query, model, index, chunks)
        answer = answer_with_llm(query, context)
        print("\n=== HASIL ===")
        print(answer)

if __name__ == "__main__":
    main()