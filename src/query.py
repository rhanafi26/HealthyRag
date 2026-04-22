"""
=============================================================
PIPELINE QUERY — RAG UTS Data Engineering
=============================================================

Pipeline ini dijalankan setiap kali user mengajukan pertanyaan:
1. Ubah pertanyaan user ke vektor (query embedding)
2. Cari chunk paling relevan dari vector database (retrieval)
3. Gabungkan konteks + pertanyaan ke dalam prompt
4. Kirim ke LLM untuk mendapatkan jawaban

Jalankan CLI dengan: python src/query.py
=============================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K         = int(os.getenv("TOP_K", 15))
VS_DIR        = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
LLM_MODEL     = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")


# =============================================================
# TODO MAHASISWA:
# Pilih implementasi yang sesuai dengan pilihan LLM kalian
# =============================================================

def load_vectorstore():
    """Load FAISS + chunks."""
    import faiss
    import json

    if not VS_DIR.exists():
        raise FileNotFoundError(
            f"Vector store tidak ditemukan di '{VS_DIR}'.\n"
            "Jalankan dulu: python src/indexing.py"
        )

    index = faiss.read_index(str(VS_DIR / "index.faiss"))

    with open(VS_DIR / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # load model sama dengan indexing
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    return {
        "index": index,
        "chunks": chunks,
        "model": model
    }

def retrieve_context(vectorstore, question: str, top_k: int = TOP_K) -> list:
    import faiss

    index = vectorstore["index"]
    chunks = vectorstore["chunks"]
    model = vectorstore["model"]

    # encode query
    query_vec = model.encode([question])

    # normalize
    faiss.normalize_L2(query_vec)

    # search
    D, I = index.search(query_vec.astype("float32"), top_k)

    contexts = []
    for i, idx in enumerate(I[0]):
        if idx < len(chunks):
            contexts.append({
                "content": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "score": round(float(D[0][i]), 4)
            })

    return contexts

def build_prompt(question: str, contexts: list) -> str:
    """
    LANGKAH 3: Membangun prompt untuk LLM.
    
    Prompt yang baik untuk RAG harus:
    - Memberikan instruksi jelas ke LLM
    - Menyertakan konteks yang sudah diambil
    - Menanyakan pertanyaan user
    - Meminta LLM untuk jujur jika tidak tahu
    
    TODO: Modifikasi prompt ini sesuai domain dan bahasa proyek kalian!
    """
    context_text = "\n\n---\n\n".join(
        [f"[Sumber: {c['source']}]\n{c['content']}" for c in contexts]
    )

    prompt = f"""Kamu adalah asisten yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.

INSTRUKSI:
- Jawab HANYA berdasarkan konteks di bawah ini
- Jika jawaban tidak ada dalam konteks, katakan "Saya tidak menemukan informasi tersebut dalam dokumen yang tersedia"
- Jawab dalam Bahasa Indonesia yang jelas dan ringkas
- Jangan mengarang informasi yang tidak ada di konteks

KONTEKS:
{context_text}

PERTANYAAN:
{question}

JAWABAN:"""
    
    return prompt


# ─────────────────────────────────────────────────────────────
# OPSI LLM A: Groq (gratis, cepat) — REKOMENDASI
# ─────────────────────────────────────────────────────────────
def get_answer_groq(prompt: str) -> str:
    """Menggunakan Groq API (gratis, sangat cepat)."""
    from groq import Groq
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,   # Rendah = jawaban lebih konsisten/faktual
        max_tokens=1024
    )
    return response.choices[0].message.content

def answer_question(question: str, vectorstore=None) -> dict:
    """
    Fungsi utama: menerima pertanyaan, mengembalikan jawaban + konteks.
    
    Returns:
        dict dengan keys: answer, contexts, prompt
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    # Retrieve
    print(f"🔍 Mencari konteks relevan untuk: '{question}'")
    contexts = retrieve_context(vectorstore, question)
    print(f"   ✅ {len(contexts)} chunk relevan ditemukan")
    
    # Build prompt
    prompt = build_prompt(question, contexts)
    
    # Generate answer
    print("🤖 Mengirim ke LLM...")

    answer = get_answer_groq(prompt)
    
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "prompt": prompt
    }


# ─── CLI Interface ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🤖 RAG System — UTS Data Engineering")
    print("  Ketik 'keluar' untuk mengakhiri")
    print("=" * 55)

    print("Memuat Database, loading..")
    try:
        vs = load_vectorstore()
        print("✅ Vector database berhasil dimuat\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)

    while True:
        print()
        question = input("❓ Pertanyaan Anda: ").strip()
        
        if question.lower() in ["keluar", "exit", "quit", "q"]:
            print("👋 Sampai jumpa!")
            break
        
        if not question:
            print("⚠️  Pertanyaan tidak boleh kosong.")
            continue
        
        try:
            result = answer_question(question, vs)

            print("\n" + "─" * 55)
            print("💬 JAWABAN:")
            print(result["answer"])
            
            print("\n📚 SUMBER KONTEKS:")
            for i, ctx in enumerate(result["contexts"], 1):
                print(f"  [{i}] Skor: {ctx['score']:.4f} | {ctx['source']}")
                print(f"      {ctx['content'][:100]}...")
            print("─" * 55)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Pastikan API key sudah diatur di file .env")
