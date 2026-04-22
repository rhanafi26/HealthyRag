"""
=============================================================
PIPELINE INDEXING — RAG UTS Data Engineering
=============================================================

Pipeline ini dijalankan SEKALI untuk:
1. Memuat dokumen dari folder data/
2. Memecah dokumen menjadi chunk-chunk kecil
3. Mengubah setiap chunk menjadi vektor (embedding)
4. Menyimpan vektor ke dalam vector database

Jalankan dengan: python src/indexing.py
=============================================================
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ─── LANGKAH 0: Load konfigurasi dari .env ───────────────────────────────────
load_dotenv()

# Konfigurasi — bisa diubah sesuai kebutuhan
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 800)) # pemilihan 800 agar bisa menangkap konteks lebih lengkap
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50)) # pemilihan 50 agar lebih efisien
DATA_DIR      = Path(os.getenv("DATA_DIR", "./data"))
VS_DIR        = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))


# ─────────────────────────────────────────────────────────────
# IMPLEMENTASI B: From Scratch (tanpa LangChain)
# Uncomment blok ini jika memilih opsi from scratch
# ─────────────────────────────────────────────────────────────

def build_index_scratch():
    """Implementasi RAG dari scratch menggunakan sentence-transformers + FAISS."""
    import json
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    import pandas as pd
    from pypdf import PdfReader

    print(" Memulai Pipeline Indexing (From Scratch)")

    # Load dokumen
    documents = []

    # load csv file
    for file_path in DATA_DIR.glob("**/*.csv"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = pd.read_csv(f)
            for _, row in content.iterrows():
                text = " ".join([f"{col}: {row[col]}" for col in content.columns])
                documents.append({
                    "source": str(file_path),
                    "content": text
                })

    # load pdf file
    for file_path in DATA_DIR.glob("**/*.pdf"):
        reader = PdfReader(file_path)
        content = ""

        for page in reader.pages:
            text += page.extract_text()
            if text:
                text = text.replace("\n", " ").strip()
                content += text + " "

        documents.append({
            "source": str(file_path),
            "content": content
        })

    print(f" {len(documents)} dokumen dimuat")

    # Chunking manual
    chunks = []
    for doc in documents:
        text = doc["content"]
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i + CHUNK_SIZE]
            if len(chunk_text) > 50:
                chunks.append({"source": doc["source"], "text": chunk_text, "id": len(chunks)})
    print(f" {len(chunks)} chunk dibuat")

    # Embedding
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f" Embedding selesai, dimensi: {embeddings.shape}")

    # Simpan ke FAISS
    VS_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(VS_DIR / "index.faiss"))

    # Simpan metadata
    with open(VS_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f" Index FAISS tersimpan di {VS_DIR}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Atau jika from scratch:
    build_index_scratch()
