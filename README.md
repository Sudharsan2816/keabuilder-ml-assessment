# KeaBuilder ML Engineer Assessment
**Dream Reflection Media — ML Engineer Role**

---

## What's Built

| Demo | Description | Run Port |
|------|-------------|----------|
| `demo1_similarity_search/` | Text similarity search API (FastAPI + TF-IDF) | 8002 |
| `demo2_schema_design/` | PostgreSQL schema for ML inputs + predictions | — |
| `docs/` | Full ML system design answers (all 7 questions) | — |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/Sudharsan2816/keabuilder-ml-assessment
cd keabuilder-ml-assessment
pip install -r requirements.txt

# Run similarity search API
cd demo1_similarity_search
python -m uvicorn app:app --reload --port 8002
```

Open: http://localhost:8002/docs

---

## Test the API

```bash
# Find similar leads
curl -X POST http://localhost:8002/find-similar \
  -H "Content-Type: application/json" \
  -d '{"query": "I want to sell my coaching program online", "top_k": 3, "corpus_type": "leads"}'

# Find similar prompts
curl -X POST http://localhost:8002/find-similar \
  -H "Content-Type: application/json" \
  -d '{"query": "write ad copy for fitness coaching", "top_k": 2, "corpus_type": "prompts"}'

# View full corpus
curl http://localhost:8002/corpus/leads
curl http://localhost:8002/corpus/prompts

# Current vs production architecture
curl http://localhost:8002/architecture
```

---

## Interactive API Docs
http://localhost:8002/docs

---

## Architecture Comparison
`GET /architecture` returns a live JSON comparison of:
- **Current:** TF-IDF cosine similarity (zero-GPU, instant startup)
- **Production:** sentence-transformers + pgvector (semantic search)
- **Face similarity:** InsightFace ArcFace embeddings + pgvector

---

## ML System Design Answers
All 7 questions answered in full → [`docs/ml_system_design_answers.md`](docs/ml_system_design_answers.md)

| Question | Topic |
|----------|-------|
| Q1 | Similarity search — current implementation + production upgrade |
| Q2 | Serving Python ML model with Node.js backend |
| Q3 | Database schema design (users, inputs, predictions, embeddings) |
| Q4 | Handling slow ML responses in UI (optimistic UI, polling, streaming) |
| Q5 | Notebook → production migration challenges |
| Q6 | LoRA for face consistency (DreamBooth-SDXL, training params, pricing) |
| Q7 | Tools and frameworks |

---

## Schema Design
Full PostgreSQL schema → [`demo2_schema_design/schema.sql`](demo2_schema_design/schema.sql)

Design rationale → [`demo2_schema_design/schema_explained.md`](demo2_schema_design/schema_explained.md)

Sample analytics queries → [`demo2_schema_design/sample_queries.sql`](demo2_schema_design/sample_queries.sql)

---

## Project Structure

```
keabuilder-ml-assessment/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── demo1_similarity_search/
│   ├── app.py                  # FastAPI app + TF-IDF search
│   ├── models.py               # Pydantic input/output models
│   ├── data.py                 # Leads + prompts corpus + architecture data
│   ├── requirements.txt
│   └── sample_output.json      # 3 test cases with expected outputs
├── demo2_schema_design/
│   ├── schema.sql              # Full PostgreSQL + pgvector schema
│   ├── schema_explained.md     # Design rationale for every decision
│   └── sample_queries.sql      # Analytics + performance queries
└── docs/
    └── ml_system_design_answers.md  # All 7 questions answered
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + uvicorn |
| ML | scikit-learn TF-IDF + cosine similarity |
| Database | PostgreSQL + pgvector |
| Production ML | sentence-transformers/all-MiniLM-L6-v2 |
| Runtime | Python 3.12 |
| Validation | Pydantic v2 |
| Production Queue | SQS / BullMQ |
| Production Cache | Redis |
