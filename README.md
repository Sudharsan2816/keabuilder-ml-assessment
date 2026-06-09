# Semantic Similarity Search API

FastAPI ML service for text similarity search, with a PostgreSQL/pgvector schema for production semantic search.

This repository was previously named `keabuilder-ml-assessment`. The recruiter-friendly name should be `semantic-similarity-search-api` because the core value is an ML search API plus production schema design, not the assessment context.

## What It Builds

### Similarity Search API

The API compares a user query against curated corpora and returns ranked matches.

Current implementation:

- TF-IDF vectorization with scikit-learn.
- Cosine similarity ranking.
- Two corpus types: leads and prompts.
- FastAPI endpoints with Pydantic validation.
- Test coverage for happy paths and validation behavior.

Production upgrade path:

- Replace TF-IDF with sentence-transformer embeddings.
- Store embeddings in PostgreSQL with pgvector.
- Add approximate nearest-neighbor indexes.
- Track model versions, latency, confidence, and prediction metadata.

### Database Schema

The repo includes a PostgreSQL schema for storing:

- Users.
- User inputs.
- Model predictions.
- Embeddings with pgvector.
- Analytics and similarity search indexes.

## Architecture

```text
Client
  -> FastAPI
  -> request validation
  -> corpus loader
  -> TF-IDF vectorizer
  -> cosine similarity
  -> ranked matches

Production target:
Client
  -> API
  -> embedding model
  -> PostgreSQL + pgvector
  -> ranked semantic matches
  -> prediction logging
```

## Tech Stack

- Python, FastAPI, Pydantic
- scikit-learn TF-IDF and cosine similarity
- PostgreSQL schema design
- pgvector production design
- pytest test suite

## Run Locally

```bash
git clone https://github.com/Sudharsan2816/semantic-similarity-search-api
cd semantic-similarity-search-api
pip install -r requirements.txt

cd demo1_similarity_search
python -m uvicorn app:app --reload --port 8002
```

Open API docs at http://localhost:8002/docs.

## Example Requests

```bash
curl -X POST http://localhost:8002/find-similar \
  -H "Content-Type: application/json" \
  -d '{"query": "I want to sell my coaching program online", "top_k": 3, "corpus_type": "leads"}'
```

```bash
curl -X POST http://localhost:8002/find-similar \
  -H "Content-Type: application/json" \
  -d '{"query": "write ad copy for fitness coaching", "top_k": 2, "corpus_type": "prompts"}'
```

## Test

```bash
cd demo1_similarity_search
python -m pytest test_app.py -q
```

## Schema Files

- `demo2_schema_design/schema.sql`
- `demo2_schema_design/schema_explained.md`
- `demo2_schema_design/sample_queries.sql`

## Portfolio Value

This repo demonstrates:

- ML API design beyond notebooks.
- Ranking and similarity search fundamentals.
- API validation and test coverage.
- Database schema design for ML inputs, predictions, and embeddings.
- A clear migration path from baseline lexical search to semantic vector search.

## Current Production Gaps

- Implement the sentence-transformers + pgvector version, not only document it.
- Add Docker Compose for API plus PostgreSQL.
- Add model/version metadata to API responses.
- Add search quality metrics such as recall@k and MRR.
- Add CI to run tests automatically.

## Recommended GitHub Metadata

- Repository name: `semantic-similarity-search-api`
- Description: `FastAPI similarity search service with TF-IDF baseline, pytest coverage, PostgreSQL schema design, and a documented sentence-transformers + pgvector upgrade path.`
- Topics: `python`, `fastapi`, `machine-learning`, `similarity-search`, `tfidf`, `scikit-learn`, `postgresql`, `pgvector`, `pytest`, `ml-engineering`
