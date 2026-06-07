# Doc-QA — Document Question Answering System

A production-grade **Retrieval-Augmented Generation (RAG)** API built with Django. Upload Word documents, ask questions in plain English, get answers grounded strictly in your documents — with full source attribution.

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [How It Actually Works — The Full Picture](#how-it-actually-works)
3. [The Math Behind Everything](#the-math-behind-everything)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
   - [Option A — Run locally (no Docker)](#option-a--run-locally-no-docker)
   - [Option B — Run with Docker](#option-b--run-with-docker)
6. [Admin Panel](#admin-panel)
7. [API Reference & Usage](#api-reference--usage)
8. [Sample Documents](#sample-documents)
9. [End-to-End Example](#end-to-end-example)
10. [Configuration](#configuration)
11. [What's Next](#whats-next)

---

## What This Is

Most LLMs like GPT-4 are trained on general internet data. They know a lot, but they don't know *your* documents — your contracts, research papers, internal reports, manuals. If you ask them about your content, they either hallucinate or admit they don't know.

**RAG (Retrieval-Augmented Generation)** solves this by giving the LLM exactly the right pieces of your document at query time, so it can answer questions grounded in real content rather than guessing.

This project implements a full RAG pipeline with two major upgrades over naive implementations:

- **Semantic + Structure-Aware Chunking** — understands document structure instead of blindly cutting every 500 words
- **Hybrid Search (Dense + BM25 + RRF)** — combines semantic understanding with keyword matching for dramatically better retrieval

---

## How It Actually Works

There are two completely separate phases. Understanding this separation is the key to understanding the whole system.

---

### Phase 1 — Indexing (happens once, when you upload a document)

When you POST a document to the API, the following pipeline runs automatically:

```
.docx file
    │
    ▼
[1] Text + Structure Extraction
    Read every paragraph, tag each one as either
    'heading' or 'text' using the .docx style metadata.
    │
    ▼
[2] Smart Chunking
    Walk through paragraphs one by one.
    At each boundary, ask three questions:
      • Is the next paragraph a heading?     → split (structural signal)
      • Did the meaning just shift?          → split (semantic signal)
      • Is the chunk already 600+ words?     → split (size safety net)
    Result: a list of coherent text chunks, each about one topic.
    │
    ▼
[3] Embedding (Dense Vectors)
    Pass each chunk through all-MiniLM-L6-v2.
    Output: a 384-dimensional vector per chunk.
    Similar meaning → similar vector → close in vector space.
    │
    ▼
[4] BM25 Indexing (Sparse / Keyword)
    Tokenize each chunk, count term frequencies.
    Store the term-frequency map per chunk.
    This is the keyword index used by BM25 at query time.
    │
    ▼
[5] Persist to Database
    Save chunk text + embedding vector + BM25 term frequencies
    to SQLite. The document is now fully indexed.
```

This happens **once per document**, triggered automatically by Django's `post_save` signal. You never call it manually.

---

### Phase 2 — Querying (happens every time you ask a question)

```
User question: "When does the contract expire?"
    │
    ▼
[1] Embed the question
    Pass it through the same MiniLM model.
    Output: a 384-dim vector representing the question's meaning.
    │
    ▼
[2] Dense Retrieval
    Compute cosine similarity between the question vector
    and every chunk vector stored in the database.
    Sort chunks by score → dense ranking list.
    │
    ▼
[3] BM25 Retrieval
    Tokenize the question: ["when", "does", "contract", "expire"]
    Score every chunk using the BM25 formula.
    Sort chunks by score → BM25 ranking list.
    │
    ▼
[4] Hybrid Fusion (RRF)
    Combine both ranking lists using Reciprocal Rank Fusion.
    Score = 1/(60 + rank_dense) + 1/(60 + rank_bm25)
    The chunks that appear near the top in BOTH lists win.
    Sort by fused score → final ranking.
    │
    ▼
[5] Build Context
    Take the top 4 chunks from the fused ranking.
    Format them as a context block:
    [From: Contract.docx, section 3]
    The agreement terminates on December 31, 2025...
    │
    ▼
[6] LLM Generation
    Send a structured prompt to the LLM via OpenRouter:
      SYSTEM: Answer only from the provided context.
              If the answer isn't there, say so.
      USER:   Context: {top 4 chunks}
              Question: {user's question}
    │
    ▼
[7] Return Answer
    Save the Q&A to history, return JSON with:
    - answer text
    - source document titles
    - timestamp
```

---

## The Math Behind Everything

### Cosine Similarity (Dense Retrieval)

Every chunk and every question gets encoded as a vector in 384-dimensional space. To find how semantically similar two texts are, we measure the angle between their vectors — not the distance. This is cosine similarity:

```
                  A · B
cos(θ) =  ─────────────────
            ‖A‖ × ‖B‖
```

Where:
- `A · B` is the dot product: Σ(Aᵢ × Bᵢ) for all 384 dimensions
- `‖A‖` is the magnitude (length) of vector A: √(Σ Aᵢ²)
- `‖B‖` is the magnitude of vector B

**Why angle and not distance?**
Distance penalizes long documents just for being long — they produce larger vectors. Cosine similarity normalizes by magnitude, so a short chunk and a long chunk can be equally relevant to a question as long as they talk about the same thing.

Result range:
```
 1.0  →  identical meaning
 0.0  →  completely unrelated
-1.0  →  opposite meaning (rare in practice)
```

In code:
```python
dot_product = np.dot(question_vec, chunk_vec)
similarity  = dot_product / (np.linalg.norm(question_vec) * np.linalg.norm(chunk_vec))
```

---

### BM25 (Keyword Retrieval)

BM25 (Best Match 25) is the classical information retrieval formula. It scores how relevant a chunk `d` is to a query by summing contributions from each query term `t`:

```
              N - df(t) + 0.5              tf(t,d) × (k₁ + 1)
score(d,q) = Σ  log( ─────────────────── + 1 ) × ──────────────────────────────────
              t      df(t) + 0.5               tf(t,d) + k₁ × (1 - b + b × |d|/avgdl)
```

Breaking down every symbol:

| Symbol | Meaning |
|--------|---------|
| `N` | Total number of chunks in the database |
| `df(t)` | How many chunks contain term `t` (document frequency) |
| `tf(t,d)` | How many times term `t` appears in chunk `d` (term frequency) |
| `\|d\|` | Length of chunk `d` in tokens |
| `avgdl` | Average chunk length across all chunks |
| `k₁ = 1.5` | Term frequency saturation — higher means more weight to frequency |
| `b = 0.75` | Length normalization — 1.0 = full normalization, 0 = none |

**The IDF part** (left of the ×):
```
      N - df(t) + 0.5
log( ─────────────── + 1 )
      df(t) + 0.5
```
Words that appear in every chunk ("the", "is", "a") get IDF ≈ 0. Words that appear in only 2 out of 500 chunks ("arbitration", "photosynthesis") get high IDF — they're discriminating.

**The TF_norm part** (right of the ×):
```
tf(t,d) × (k₁ + 1)
────────────────────────────────────────
tf(t,d) + k₁ × (1 - b + b × |d|/avgdl)
```
A chunk that says "contract" 5 times in 100 words is more relevant than one that says it 5 times in 2000 words. The `b` parameter controls how aggressively we penalize long chunks.

---

### Semantic Chunking (Cosine at Paragraph Level)

During indexing, we embed every paragraph with MiniLM and compare adjacent paragraphs:

```
similarity(paraᵢ, paraᵢ₊₁) = cosine(embed(paraᵢ), embed(paraᵢ₊₁))
```

If `similarity < 0.45` → the meaning shifted → split here.

```python
SEMANTIC_SPLIT_THRESHOLD = 0.45  # in processing.py — tune this freely
MAX_CHUNK_WORDS = 600            # hard cap regardless of similarity
```

---

### Reciprocal Rank Fusion (Hybrid Search Fusion)

RRF works on ranks, not raw scores — so the scale difference between cosine (0–1) and BM25 (0–∞) doesn't matter.

```
           1                    1
RRF(d) = ─────────────── + ───────────────
          K + rank_dense    K + rank_bm25

K = 60  (standard constant)
```

Example with 3 chunks:
```
Dense ranking:   A=1,  B=2,  C=3
BM25  ranking:   B=1,  A=2,  C=3

RRF(A) = 1/(60+1) + 1/(60+2) = 0.03252
RRF(B) = 1/(60+2) + 1/(60+1) = 0.03252
RRF(C) = 1/(60+3) + 1/(60+3) = 0.03175
```

A and B both appear near the top in both lists → tied. C is last in both → lowest score.

---

## Project Structure

```
Doc-QA/
│
├── core/                        # Django project config
│   ├── settings.py              # DB, installed apps, OpenRouter keys
│   ├── urls.py                  # Root URL routing
│   └── wsgi.py
│
├── documents/                   # Ingestion app
│   ├── models.py                # Document + DocumentChunk (embeddings + BM25)
│   ├── processing.py            # Full indexing pipeline
│   ├── signals.py               # Triggers processing on upload
│   ├── views.py                 # DocumentViewSet (full CRUD)
│   └── urls.py                  # /api/documents/
│
├── qa/                          # Query app
│   ├── models.py                # QAHistory
│   ├── pipeline.py              # Hybrid search + LLM generation
│   ├── admin.py                 # Admin panel with Ask a Question page
│   ├── views.py                 # AskView + HistoryView
│   └── urls.py                  # /api/ask/ and /api/history/
│
├── templates/                   # Django Admin custom templates
│   └── admin/qa/qahistory/
│       ├── ask.html             # Ask a Question page
│       └── change_list.html     # Adds Ask button to history list
│
├── sample-Docs/                 # Sample .docx files for testing
│
├── Dockerfile                   # Container build instructions
├── docker-compose.yml           # Container orchestration
├── .dockerignore                # Files excluded from Docker image
├── .env.example                 # Copy to .env and fill in your keys
├── requirements.txt
└── manage.py
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+ (for local run)
- Docker Desktop (for Docker run)
- An [OpenRouter](https://openrouter.ai) API key

### Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
SECRET_KEY=replace-this-with-a-long-random-string
DEBUG=True
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=mistralai/mistral-7b-instruct
```

---

### Option A — Run Locally (no Docker)

**Step 1 — Create and activate a virtual environment:**

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

**Step 2 — Install dependencies:**

```bash
pip install -r requirements.txt
```

The first run downloads `all-MiniLM-L6-v2` (~90MB) automatically from HuggingFace. This happens once.

**Step 3 — Run migrations:**

```bash
python manage.py migrate
```

**Step 4 — Create an admin user:**

```bash
python manage.py createsuperuser
```

**Step 5 — Start the server:**

```bash
python manage.py runserver
```

The app is now running at `http://localhost:8000`.

**To stop:** `Ctrl+C`

**To start again:** just run `python manage.py runserver` — your data persists in `db.sqlite3`.

---

### Option B — Run with Docker

Docker packages the entire app — Python, dependencies, and the embedding model — into a self-contained container. No manual setup needed beyond Docker Desktop.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running (you'll see "Engine running" in the bottom left).

**Step 1 — Build and start:**

```bash
docker compose up --build
```

The first build takes 5–15 minutes — it installs all packages and downloads the MiniLM model inside the image. Every subsequent start is instant.

You'll know it's ready when you see:
```
web-1  | Starting development server at http://0.0.0.0:8000/
```

**Step 2 — Create an admin user** (open a new terminal, keep the first one running):

```bash
docker compose exec web python manage.py createsuperuser
```

**The app is now running at `http://localhost:8000`.**

---

**Docker commands reference:**

| Command | What it does |
|---------|-------------|
| `docker compose up --build` | Build image and start (first time or after code changes) |
| `docker compose up` | Start normally — fast, no rebuild, data preserved |
| `docker compose down` | Stop the container, keep all data |
| `docker compose down -v` | Stop and **wipe all data** (fresh start) |
| `docker compose exec web python manage.py createsuperuser` | Create admin user |
| `docker compose logs -f` | Watch live server logs |

> **Data persistence:** Uploaded documents and the database survive `docker compose down` and `docker compose up` via named Docker volumes. Use `docker compose down -v` only when you want a completely clean slate.

---

## Admin Panel

The Django Admin panel at `http://localhost:8000/admin/` gives you a full UI to manage everything without touching the API.

**Login:** use the superuser credentials you created above.

**What you can do:**

**Documents section:**
- View all uploaded documents with chunk counts
- See every chunk a document was split into
- Delete documents

**QA History section:**
- View all past questions and answers with their source documents
- **Ask a Question** — a blue button in the top right of the history list. Click it, type your question in plain text, hit Ask. The full RAG pipeline runs and the answer appears on the same page. No JSON, no curl, no Postman needed.

---

## API Reference & Usage

### Upload a Document

```
POST /api/documents/
Content-Type: multipart/form-data
```

```bash
curl -X POST http://localhost:8000/api/documents/ \
  -F "title=Employment Contract" \
  -F "file=@/path/to/contract.docx"
```

**Response (201 Created):**
```json
{
    "id": 1,
    "title": "Employment Contract",
    "extracted_text": "This Employment Agreement...",
    "uploaded_at": "2026-06-07T10:00:00Z",
    "chunk_count": 12
}
```

---

### List / Get / Delete Documents

```bash
curl http://localhost:8000/api/documents/          # list all
curl http://localhost:8000/api/documents/1/        # get one
curl -X DELETE http://localhost:8000/api/documents/1/  # delete
```

---

### Ask a Question

```
POST /api/ask/
Content-Type: application/json
```

```bash
curl -X POST http://localhost:8000/api/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?"}'
```

**Response (201 Created):**
```json
{
    "id": 1,
    "question": "What are the payment terms?",
    "answer": "Payment is due within 30 days of invoice.",
    "sources": ["Employment Contract"],
    "created_at": "2026-06-07T10:05:00Z"
}
```

---

### View Question History

```bash
curl http://localhost:8000/api/history/
```

Returns all past Q&As, newest first.

---

## Sample Documents

The `sample-Docs/` folder contains ready-to-use `.docx` files for testing. Upload one and start asking questions immediately without needing to prepare your own documents.

---

## End-to-End Example

**1. Start the server** (Docker or local — your choice):
```bash
docker compose up          # Docker
# or
python manage.py runserver # local
```

**2. Upload a sample document:**
```bash
curl -X POST http://localhost:8000/api/documents/ \
  -F "title=World History 20th Century" \
  -F "file=@sample-Docs/history.docx"
```

Server console output:
```
Processing: World History 20th Century
  Extracted 47 paragraphs (18432 chars)
  [Chunker] Semantic split at para 8 (sim=0.31)
  [Chunker] Semantic split at para 19 (sim=0.28)
  Created 9 smart chunks
  Done! Saved 9 chunks with embeddings + BM25 stats
```

**3. Ask a question:**
```bash
curl -X POST http://localhost:8000/api/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "When did World War 1 start?"}'
```

Response:
```json
{
    "id": 1,
    "question": "When did World War 1 start?",
    "answer": "World War I started on July 28, 1914.",
    "sources": ["World History 20th Century"],
    "created_at": "2026-06-07T10:30:34Z"
}
```

**4. Or ask via Admin panel:**
Go to `http://localhost:8000/admin/qa/qahistory/` → click **Ask a Question** → type → hit Ask.

---

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key | any long random string |
| `DEBUG` | Debug mode | `True` or `False` |
| `OPENROUTER_API_KEY` | Your OpenRouter API key | `sk-or-v1-...` |
| `OPENROUTER_MODEL` | Model for answer generation | `mistralai/mistral-7b-instruct` |

Some good free/cheap OpenRouter models:
```
mistralai/mistral-7b-instruct
meta-llama/llama-3-8b-instruct
google/gemma-7b-it
```

### Tunable Parameters

**`documents/processing.py`**
```python
SEMANTIC_SPLIT_THRESHOLD = 0.45  # Lower → fewer bigger chunks; Higher → more smaller chunks
MAX_CHUNK_WORDS = 600            # Hard cap per chunk
```

**`qa/pipeline.py`**
```python
K1 = 1.5    # BM25 term frequency weight
B  = 0.75   # BM25 length normalization
RRF_K = 60  # Fusion constant
```



## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Django 4.2 + Django REST Framework |
| Embedding model | `all-MiniLM-L6-v2` via sentence-transformers |
| Keyword search | BM25 (implemented from scratch) |
| Vector fusion | Reciprocal Rank Fusion (implemented from scratch) |
| LLM provider | OpenRouter (OpenAI-compatible API) |
| LLM client | LangChain `ChatOpenAI` adapter |
| Document parsing | python-docx |
| Database | SQLite (via Django ORM) |
| Containerization | Docker + Docker Compose |

---

