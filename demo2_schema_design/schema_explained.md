# KeaBuilder ML Schema — Design Decisions

## Overview

The schema stores two things: **what users typed** (user_inputs) and **what the ML model predicted** (predictions). They are deliberately separate tables.

---

## Table: `users`

| Field | Type | Why |
|-------|------|-----|
| `id` | UUID | See UUID rationale below |
| `email` | VARCHAR(255) UNIQUE | Natural unique key for login/lookup |
| `plan` | VARCHAR(50) | Gates access to ML features (free vs pro) |
| `created_at` | TIMESTAMP | Tracks cohort for analytics |

---

## Table: `user_inputs`

Stores every raw input before any ML processing.

| Field | Type | Why |
|-------|------|-----|
| `id` | UUID | Primary key |
| `user_id` | UUID FK | Links input to the user who submitted it |
| `session_id` | UUID | Groups inputs within one browser session (no FK — session is ephemeral) |
| `input_text` | TEXT | No VARCHAR limit — user messages can be any length |
| `input_type` | VARCHAR(50) | `lead_form`, `ai_prompt`, `search_query`, `chatbot` — drives routing logic |
| `source` | VARCHAR(50) | `web_form`, `api`, `landing_page` — attribution for analytics |
| `metadata` | JSONB | See JSONB rationale below |
| `ip_address` | INET | PostgreSQL native IP type — handles both IPv4 and IPv6, enables geo-lookup |

**Why store inputs separately from predictions?**
A single input can be processed by multiple models (e.g. classifier + similarity search simultaneously). Separating them means you never duplicate the raw text, and you can add new model results to an existing input without touching the inputs table.

---

## Table: `predictions`

Stores one row per model inference. One input can have many predictions.

| Field | Type | Why |
|-------|------|-----|
| `input_id` | UUID FK | Links back to the raw input |
| `model_name` | VARCHAR(100) | `lead-classifier-v1`, `similarity-search-v1` |
| `model_version` | VARCHAR(20) | Enables A/B comparison between model versions |
| `prediction_label` | VARCHAR(100) | `HOT`, `WARM`, `COLD`, or a matched entry ID |
| `confidence` | FLOAT | 0.0–1.0 — used to flag low-confidence predictions for review |
| `raw_output` | JSONB | Full model response — preserves everything for debugging and retraining |
| `latency_ms` | INTEGER | Tracks inference time — alerts if p99 spikes |
| `status` | VARCHAR(20) | `pending`, `completed`, `failed` — supports async job patterns |
| `error_message` | TEXT | Populated only on failure — no wasted space otherwise |

**Why store `raw_output` as JSONB?**
Different models return different shapes. A classifier returns `{label, confidence, signals}`. A similarity search returns `{matches, scores}`. JSONB stores both without needing schema migrations for every model change. You can still index specific keys inside JSONB when needed.

---

## Table: `embeddings`

Stores dense vector representations for semantic similarity search.

| Field | Type | Why |
|-------|------|-----|
| `embedding` | vector(384) | 384 dimensions from `sentence-transformers/all-MiniLM-L6-v2` |
| `model_name` | VARCHAR(100) | Tracks which embedding model was used — critical for upgrades |
| `input_type` | VARCHAR(50) | `text`, `face`, `template` — separates text from face embeddings in same table |

**Why pgvector instead of a separate vector database?**
- Keeps everything in one database — no extra infrastructure
- Supports standard SQL joins (find embedding + its prediction in one query)
- IVFFlat index gives sub-100ms search on millions of vectors
- When scale demands it, migrate to Pinecone without changing the application interface

---

## Design Rationale

### Why UUID instead of integer IDs?

1. **No sequential leak** — integer IDs expose record counts (`/user/1042` tells attackers you have 1042 users)
2. **Merge-safe** — UUIDs from different databases never collide (critical for multi-region or data migrations)
3. **Pre-generatable** — the client can generate the UUID before inserting, enabling optimistic UI patterns

### Why JSONB for `metadata` and `raw_output`?

- `metadata` stores browser, device, UTM params, A/B test variant — this set changes constantly as the product evolves. Adding a JSONB column once is better than 20 schema migrations.
- `raw_output` stores the full model response — different models return different shapes. JSONB handles heterogeneous data without a schema migration per model.
- PostgreSQL JSONB is indexed and queryable: `metadata->>'utm_source' = 'facebook'` works with a GIN index.

### Why pgvector for embeddings?

- Embeddings are 384-dimensional float arrays — regular columns can't store or query them efficiently
- pgvector adds `vector` type + operators (`<=>` cosine distance, `<->` L2 distance)
- IVFFlat index reduces search from O(n) full scan to O(√n) approximate nearest neighbor
- Lives in the same PostgreSQL instance — no cross-service latency for joins

### Index Strategy

- `idx_user_inputs_user_id` — every query filters by user
- `idx_predictions_label` — HOT lead dashboards filter by label
- `idx_predictions_model` — model performance queries group by model_name + version
- `idx_embeddings_vector` (IVFFlat) — fast approximate nearest-neighbor for similarity search
- All timestamp indexes use `DESC` — most queries are "recent first"
