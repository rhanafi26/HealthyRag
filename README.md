# 🤖 RAG Starter Pack — UTS Data Engineering

> **Retrieval-Augmented Generation** — Sistem Tanya-Jawab Cerdas Berbasis Dokumen

Starter pack ini adalah **kerangka awal** proyek RAG untuk UTS Data Engineering D4.
Mahasiswa mengisi, memodifikasi, dan mengembangkan kode ini sesuai topik kelompok masing-masing.

---

## 👥 Identitas Kelompok

| Nama                            | NIM | Tugas Utama |
|---------------------------------|-----|-------------|
| Alan Herva Ikhsan Saputra       | 01  | PM          |
| Riza Hanafi                     | 26  | DE          |
| Vicky Rizkyanto                 | 30  | DA          |

**Topik Domain:** *(Kesehatan)*  
**Stack yang Dipilih:** *(LangChain)*  
**LLM yang Digunakan:** *(Groq)*  
**Vector DB yang Digunakan:** *(FAISS)*

---

## 🗂️ Struktur Proyek

```
rag-uts-[nama-kelompok]/
├── data/                    # Dokumen sumber Anda (PDF, TXT, dll.)
│   └── sample.txt           # Contoh dokumen (ganti dengan dokumen Anda)
├── src/
│   ├── indexing.py          # 🔧 WAJIB DIISI: Pipeline indexing
│   ├── query.py             # 🔧 WAJIB DIISI: Pipeline query & retrieval
│   ├── embeddings.py        # 🔧 WAJIB DIISI: Konfigurasi embedding
│   └── utils.py             # Helper functions
├── ui/
│   └── app.py               # 🔧 WAJIB DIISI: Antarmuka Streamlit
├── docs/
│   └── arsitektur.png       # 📌 Diagram arsitektur (buat sendiri)
├── evaluation/
│   └── hasil_evaluasi.xlsx  # 📌 Tabel evaluasi 10 pertanyaan
├── notebooks/
│   └── 01_demo_rag.ipynb    # Notebook demo dari hands-on session
├── .env.example             # Template environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Cara Memulai (Quickstart)

### 1. Clone & Setup

```bash
# Clone repository ini
git clone https://github.com/[username]/rag-uts-[kelompok].git
cd rag-uts-[kelompok]

# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau: venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi API Key

```bash
# Salin template env
cp .env.example .env

# Edit .env dan isi API key Anda
# JANGAN commit file .env ke GitHub!
```

### 3. Siapkan Dokumen

Letakkan dokumen sumber Anda di folder `data/`:
```bash
# Contoh: salin PDF atau TXT ke folder data
cp dokumen-saya.pdf data/
```

### 4. Jalankan Indexing (sekali saja)

```bash
python src/indexing.py
```

### 5. Jalankan Sistem RAG

```bash
# Dengan Streamlit UI
streamlit run ui/app.py

# Atau via CLI
python src/query.py
```

---

## 🔧 Konfigurasi

Semua konfigurasi utama ada di `src/config.py` (atau langsung di setiap file):

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `CHUNK_SIZE` | 500 | Ukuran setiap chunk teks (karakter) |
| `CHUNK_OVERLAP` | 50 | Overlap antar chunk |
| `TOP_K` | 3 | Jumlah dokumen relevan yang diambil |
| `MODEL_NAME` | *(isi)* | Nama model LLM yang digunakan |

---

## 📊 Hasil Evaluasi

*(Isi setelah pengujian selesai)*

| # | Pertanyaan | Jawaban Sistem | Jawaban Ideal | Skor (1-5) |
|---|-----------|----------------|---------------|-----------|
| 1 | Provinsi yang memiliki tingkat obesitas tertinggi | Provinsi DKI Jakarta memiliki tingkat obesitas tertinggi dengan nilai 32,3% (30,1 - 34,5). | Ideal | 5 |
| 2 | Berapa persentase status gizi normal di Indonesia | Saya tidak menemukan informasi tentang persentase status gizi normal di Indonesia dalam dokumen yang tersedia. | Presentasi status gizi normal di inonesia adalah n% | 1 |
| 3 | Bandingkan tingkat obesitas antara DKI Jakarta dan Jawa Barat | Tingkat obesitas di DKI Jakarta sebesar 32,3% (30,1 - 34,5) untuk laki-laki dan 59,1% (56,9 - 61,2) untuk perempuan. Sementara itu, tingkat obesitas di Jawa Barat sebesar 20,2% (19,2 - tidak ada data untuk perempuan). | Ideal | 5 |
| 4 | Siapa yang lebih tinggi obesitasnya, laki-laki atau perempuan? | Saya tidak menemukan informasi tentang perbandingan obesitas antara laki-laki dan perempuan dalam dokumen yang tersedia. | lebih tinggi n | 1 |
| 5 | Mana yang lebih tinggi obesitas perkotaan atau pedesaan | Saya tidak menemukan informasi tentang perbandingan obesitas perkotaan dan pedesaan dalam dokumen yang tersedia. | Yang lebih tinggi tingkat obesitasnya daerah n | 1 |
| 6 | Bandingkan status gizi laki laki dan perempuan | Berdasarkan data yang tersedia, tidak ada informasi yang langsung membandingkan status gizi laki-laki dan perempuan. Namun, kita dapat melihat bahwa prevalensi status gizi yang sama (misalnya, obesitas) memiliki nilai yang berbeda antara laki-laki dan perempuan di setiap provinsi.

Misalnya, di provinsi Aceh, prevalensi obesitas sentral pada laki-laki adalah 18,0% (16,7-19,4), sedangkan pada perempuan adalah 17,1% (16,6-17,7). Di provinsi Sumatera Utara, prevalensi obesitas sentral pada laki-laki adalah 26,4% (24,9-28,0), sedangkan pada perempuan adalah 17,8% (17,3-18,4).

Dengan demikian, dapat disimpulkan bahwa prevalensi status gizi pada laki-laki dan perempuan memiliki perbedaan yang signifikan di setiap provinsi. | Ideal | 5 |
| 7 | Provinsi dengan jumlah jumlah penduduk tebanyak | Provinsi dengan jumlah penduduk terbanyak adalah SULAWESI SELATAN dengan jumlah penduduk 9.362.290. | Ideal | 5 |
| 8 | Jumlah penduduk undernouris terbanyak  | Jumlah penduduk undernouris terbanyak terdapat di Provinsi DI YOGYAKARTA dengan jumlah 37224940 orang. | Ideal | 5 |
| 9 | Apa makanan tinggi kalori | Berdasarkan data yang tersedia, makanan yang tinggi kalori adalah Kacang Kedelai basah dengan kalori 286.0. | Ideal | 5 |
| 10 | Makanan pokok provinsi aceh | Saya tidak menemukan informasi tentang makanan pokok provinsi Aceh dalam dokumen yang tersedia. Dokumen yang tersedia hanya berisi data konsumsi pangan di provinsi Maluku Utara. | Provinsi n dengan makanan pokok n | 1 |


**Rata-rata Skor:** ...  
**Analisis:** ...

---

## 🏗️ Arsitektur Sistem

*(Masukkan gambar diagram arsitektur di sini)*

```
[Dokumen] → [Loader] → [Splitter] → [Embedding] → [Vector DB]
                                                         ↕
[User Query] → [Query Embed] → [Retriever] → [Prompt] → [LLM] → [Jawaban]
```

---

## 📚 Referensi & Sumber

- LLM: *(Groq)*
- Vector DB: *(FAISS docs)*
- Tutorial yang digunakan: *()*

---

## 👨‍🏫 Informasi UTS

- **Mata Kuliah:** Data Engineering
- **Program Studi:** D4 Teknologi Rekayasa Perangkat Lunak
- **Deadline:** *(23 April 2026)*
