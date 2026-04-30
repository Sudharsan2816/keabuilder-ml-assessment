# KeaBuilder ML System Design Answers

---

## Q1: Similarity Search System

### Current Implementation (Demo 1 — Working API)

The working demo lives in `demo1_similarity_search/app.py`. It uses TF-IDF cosine similarity to find the most similar entries to a user query.

**How it works:**

```
User query → TfidfVectorizer → cosine_similarity → ranked results
```

1. The corpus (10 leads + 5 prompts) is held in memory as Python lists (`data.py`)
2. On each `/find-similar` request, all texts + the query are vectorized together using `TfidfVectorizer(ngram_range=(1,2))` — bigrams capture two-word phrases like "landing page" and "follow-up emails"
3. `cosine_similarity(query_vector, corpus_matrix)` scores every entry in one matrix operation
4. Results are sorted descending and the top-k are returned

**Top match logic (`app.py` lines 33–52):**
```python
tfidf_matrix = vectorizer.fit_transform(all_texts)  # query appended last
query_vector = tfidf_matrix[-1]                      # last row = query
corpus_matrix = tfidf_matrix[:-1]                   # all other rows = corpus
scores = cosine_similarity(query_vector, corpus_matrix)[0]
```

The query is included in the same fit call so it shares the same vocabulary. This matters because TF-IDF weights are relative to the full document set.

**Dual corpus support:**
- `corpus_type=leads` → searches the leads database (10 entries)
- `corpus_type=prompts` → searches the AI prompt history (5 entries)

**Strengths of current approach:**
- Zero external dependencies beyond scikit-learn
- Instant startup — no model download
- Fully interpretable — you can see exactly which words drove the match
- Bigrams catch domain phrases ("funnel builder", "email automation")

**Weaknesses:**
- Keyword-only — "sell coaching" won't match "monetise my course" even though they mean the same thing
- No semantic understanding of synonyms or related concepts
- Score range is narrow (0.05–0.6 typical) — hard to set a meaningful threshold

### Production Upgrade Path

Replace TF-IDF with dense embeddings from `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# At startup: pre-compute corpus embeddings
corpus_embeddings = model.encode(corpus_texts)  # shape: (N, 384)

# At query time:
query_embedding = model.encode([query])          # shape: (1, 384)
scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
```

Store embeddings in PostgreSQL with pgvector:
```sql
SELECT id, text, 1 - (embedding <=> $1::vector) AS score
FROM user_inputs
ORDER BY score DESC
LIMIT 5;
```

Semantic search understands that "sell coaching" and "monetise my course" are the same intent. TF-IDF does not. Score threshold for production: > 0.7 = strong match, > 0.4 = partial match.

---

## Q2: Serving ML Model with Node.js Backend

### Architecture

```
Client
  │
  ▼
Node.js Backend (:3000)          Python ML Service (:8002)
├── Auth + JWT validation    ◄──► ├── FastAPI app
├── User sessions                 ├── Model loaded at startup
├── Business logic                ├── POST /find-similar
├── Rate limiting                 ├── POST /classify-lead
└── axios.post to ML service      └── GET /health
```

### Node.js → Python call

```javascript
const axios = require("axios");

async function findSimilarLeads(userQuery) {
  const { data } = await axios.post(
    "http://ml-service:8002/find-similar",
    {
      query: userQuery,
      top_k: 3,
      corpus_type: "leads"
    },
    { timeout: 5000 }
  );
  return data;
}
```

The ML service URL uses the Docker service name (`ml-service`) — resolved internally within the Docker network. No port exposure to the public internet.

### Deployment Options

**Option A: docker-compose (development + staging)**
```yaml
services:
  backend:
    build: ./node-backend
    ports: ["3000:3000"]
    environment:
      ML_SERVICE_URL: http://ml-service:8002

  ml-service:
    build: ./demo1_similarity_search
    ports: ["8002:8002"]  # internal only in prod
```
Both services share the same Docker network. Node calls Python by service name.

**Option B: ECS (production)**
- Node.js task and Python task in the same ECS service definition
- Communicate via `localhost` (same task) or internal ALB
- Python container never exposed to public — only Node.js has a public target group

**Option C: ONNX Runtime in Node.js**
```javascript
const ort = require("onnxruntime-node");
const session = await ort.InferenceSession.create("model.onnx");
```
Only viable for simple models (linear classifiers, small transformers). Cannot run sentence-transformers or any model with Python-specific ops. Adds 500MB+ to your Node.js image. **Not recommended** for KeaBuilder's use case.

**Why Python microservice beats ONNX in Node.js:**
- Full scikit-learn, transformers, and diffusers ecosystem available
- GPU support (CUDA) is native in Python, fragile in Node.js
- Separate scaling — ML service can scale independently based on inference load
- Model updates don't require a Node.js redeploy
- FastAPI gives you free Swagger UI for testing in development

---

## Q3: Database Schema

Full schema in `demo2_schema_design/schema.sql`. Full design rationale in `demo2_schema_design/schema_explained.md`.

### Tables

**users** — account record with plan tier (free/pro/enterprise)

**user_inputs** — every raw text input before ML processing
- `input_type`: `lead_form`, `ai_prompt`, `search_query`, `chatbot`
- `source`: `web_form`, `api`, `landing_page` (attribution)
- `metadata` JSONB: browser, device, UTM params, A/B variant

**predictions** — one row per model inference
- Decoupled from inputs — one input can have multiple predictions (classifier + similarity run simultaneously)
- `model_name` + `model_version` enable A/B testing between model versions
- `status`: `pending` → `completed` or `failed` — supports async patterns
- `raw_output` JSONB: preserves full response for debugging and retraining data collection

**embeddings** — 384-dim dense vectors for semantic search
- `vector(384)` type from pgvector extension
- IVFFlat index: `USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)`
- `input_type` separates text embeddings from face embeddings in the same table

### Key Design Decisions

| Decision | Reason |
|----------|--------|
| UUID over integer IDs | No sequential leak, merge-safe across databases |
| JSONB for metadata | Schema evolves without migrations — UTM params, device info, A/B variants |
| JSONB for raw_output | Each model returns a different shape — no schema migration per model |
| Separate predictions table | One input → many predictions; avoids duplicating raw text |
| pgvector for embeddings | SQL joins work natively; IVFFlat gives sub-100ms search at scale |
| INET for ip_address | Native IPv4/IPv6 support; enables geo-lookup |

---

## Q4: Handling Slow ML Responses in UI

ML inference can take 1–10 seconds. Users abandon after 3 seconds of nothing. The solution is making the wait feel shorter, not making it faster.

### Pattern 1: Optimistic UI

Show a placeholder result immediately, replace with real result when ready.

```javascript
// React component
function LeadResult({ query }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Show skeleton immediately
    setLoading(true);

    fetch("/api/classify-lead", {
      method: "POST",
      body: JSON.stringify({ message: query })
    })
      .then(r => r.json())
      .then(data => {
        setResult(data);
        setLoading(false);
      });
  }, [query]);

  if (loading) return <LeadResultSkeleton />;
  return <LeadResultCard result={result} />;
}
```

### Pattern 2: Progress Bar with Estimated Time

For operations with known average latency (e.g. p50 = 2.1s from your `latency_ms` column):

```javascript
function ProgressBar({ estimatedMs = 2100 }) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(p => Math.min(p + (100 / (estimatedMs / 100)), 90));
    }, 100);
    return () => clearInterval(interval);
  }, []);

  // Jump to 100 when real result arrives
  return <div style={{ width: `${progress}%` }} className="progress-bar" />;
}
```

Cap at 90% until the real response arrives — never hit 100% before the result, it feels dishonest.

### Pattern 3: Polling vs WebSocket

**Polling** (simpler, use for < 5 req/sec):
```javascript
const poll = async (jobId) => {
  const interval = setInterval(async () => {
    const { data } = await axios.get(`/api/jobs/${jobId}`);
    if (data.status === "completed") {
      clearInterval(interval);
      setResult(data.result);
    }
  }, 800);
};
```

**WebSocket** (use when > 5 req/sec or real-time required):
```javascript
const ws = new WebSocket("wss://api.keabuilder.com/ws");
ws.onmessage = (event) => {
  const { jobId, result } = JSON.parse(event.data);
  if (jobId === currentJobId) setResult(result);
};
```

Use polling for KeaBuilder's lead classification (low volume, simple). Use WebSocket for live chat or real-time analytics dashboards.

### Pattern 4: Streaming for LLMs

When Claude generates a response, stream tokens directly to the UI:

```javascript
const response = await fetch("/api/generate", { method: "POST", body: ... });
const reader = response.body.getReader();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = new TextDecoder().decode(value);
  setPartialResponse(prev => prev + chunk);
}
```

The server uses `StreamingResponse` (FastAPI) or `Transfer-Encoding: chunked` (Node.js). The user sees words appearing in real time — perceived wait drops to near zero.

### Pattern 5: Error State + Retry

```javascript
if (error) {
  return (
    <div className="error-state">
      <p>Classification failed. Please try again.</p>
      <button onClick={retry}>Retry</button>
    </div>
  );
}
```

Always give the user an exit. Log the error with the `input_id` so you can debug from the `predictions` table where `status = 'failed'`.

---

## Q5: Notebook to Production Challenges

### Challenge 1: Dependency Versioning

**Problem:** A Jupyter notebook runs with whatever is installed in the analyst's environment. Six months later, `scikit-learn` updates and `TfidfVectorizer`'s default tokenizer changes — the model now returns different results silently.

**Solution: Docker + pinned requirements**

```dockerfile
FROM python:3.12.3-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt   # exact versions pinned
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
```

`requirements.txt`:
```
scikit-learn==1.4.2
numpy==1.26.4
sentence-transformers==2.7.0
```

The Docker image is immutable. The same container that passed QA runs in production. Never use `pip install scikit-learn` without a version pin in production.

### Challenge 2: Data Validation Gap

**Problem:** A notebook assumes the input is always a clean string. In production, users send empty strings, 10,000-word essays, SQL injection attempts, and None values. The notebook has no validation — it silently produces wrong results or crashes.

**Solution: Pydantic schemas at the API boundary**

```python
from pydantic import BaseModel, field_validator

class QueryInput(BaseModel):
    query: str
    top_k: int = 3

    @field_validator("query")
    @classmethod
    def query_must_be_valid(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters")
        if len(v) > 2000:
            raise ValueError("Query too long — max 2000 characters")
        return v.strip()
```

Pydantic validates, coerces, and documents the input schema automatically. Invalid inputs return a 422 with a human-readable error before they reach the model.

### Challenge 3: No Monitoring or Drift Detection

**Problem:** A notebook produces results. Nobody checks if the results are getting worse over time. In production, data distribution shifts — users start asking questions the model was never trained on — and the model silently degrades.

**Solution: Prediction logging + alerting**

Every inference writes to the `predictions` table:
```python
db.execute("""
    INSERT INTO predictions (input_id, model_name, model_version,
        prediction_label, confidence, latency_ms, status)
    VALUES ($1, $2, $3, $4, $5, $6, 'completed')
""", input_id, "lead-classifier-v1", "1.0.0", label, confidence, latency_ms)
```

Monitoring queries run on a schedule:
```sql
-- Alert if avg confidence drops below 0.6 (model is uncertain)
SELECT AVG(confidence) FROM predictions
WHERE model_name = 'lead-classifier-v1'
  AND created_at > NOW() - INTERVAL '1 hour';

-- Alert if error rate exceeds 5%
SELECT COUNT(*) FILTER (WHERE status = 'failed') * 100.0 / COUNT(*)
FROM predictions WHERE created_at > NOW() - INTERVAL '1 hour';
```

Set up PagerDuty or Slack alerts when these thresholds are breached. Retrain the model when confidence trends downward over a 7-day window.

---

## Q6: LoRA for Face Consistency

KeaBuilder needs AI-generated images of a person to look consistent across different scenes (e.g. the same founder in 10 different ad creatives). DreamBooth-LoRA is the right approach.

### What is LoRA?

LoRA (Low-Rank Adaptation) fine-tunes a diffusion model by adding small adapter matrices to the attention layers. Instead of retraining 800M parameters, you train ~2M. The result: the model learns a specific person's face in ~20 minutes on a single GPU.

### Training Pipeline

**Requirements:**
- 10–20 photos of the person (different angles, lighting, backgrounds)
- A100 GPU (Google Colab Pro+ ~$10, or AWS p3.2xlarge)
- Base model: SDXL 1.0

**Training parameters:**
```python
training_config = {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "instance_prompt": "a photo of sks person",
    "num_train_steps": 1000,
    "learning_rate": 1e-4,
    "lora_rank": 16,         # higher = more capacity, more VRAM
    "lora_alpha": 32,        # scaling factor = alpha/rank = 2.0
    "train_batch_size": 1,
    "gradient_accumulation": 4
}
```

**Training command (diffusers):**
```bash
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="./user_photos/" \
  --instance_prompt="a photo of sks person" \
  --output_dir="./lora_output/" \
  --rank=16 \
  --num_train_epochs=50 \
  --learning_rate=1e-4
```

### Inference Code

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("s3://keabuilder-loras/user-123/lora_weights.safetensors")
pipe.to("cuda")

image = pipe(
    prompt="a photo of sks person presenting at a conference, professional lighting",
    negative_prompt="blurry, distorted face, low quality",
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("output.png")
```

### Storage Per User

Each LoRA file is ~50MB (safetensors format). Store in S3:
```
s3://keabuilder-loras/
  user-{uuid}/
    lora_weights.safetensors   # ~50MB
    training_metadata.json     # steps, date, base model version
    preview_images/            # 4 test images generated at training end
```

### LoRA Rank Trade-offs

| Rank | File Size | Training Time | Face Quality | VRAM Required |
|------|-----------|---------------|--------------|---------------|
| 4 | ~12MB | 8 min | Good | 12GB |
| 16 | ~48MB | 20 min | Very Good | 16GB |
| 32 | ~96MB | 40 min | Excellent | 24GB |
| 64 | ~192MB | 80 min | Excellent | 40GB |

**Recommendation for KeaBuilder:** rank=16 hits the best cost-quality balance. Rank 4 is too low for face consistency. Rank 32+ is overkill for ad creative use.

### Cost Estimate

| Option | Cost Per Training Run | Latency |
|--------|----------------------|---------|
| Google Colab A100 | ~$2–4 | 20 min |
| AWS p3.2xlarge (V100) | ~$3.06/hr → ~$1 per run | 20 min |
| Replicate API (managed) | ~$0.50–2 per run | 15 min |
| RunPod H100 | ~$3.89/hr → ~$1.30 per run | 8 min |

For a SaaS at scale: use Replicate or a dedicated GPU pool. Train once per user, cache the LoRA in S3. Inference (image generation) is ~$0.01–0.05 per image.

---

## Q7: Tools and Frameworks Used

### Demo 1 — Similarity Search API
| Tool | Purpose |
|------|---------|
| FastAPI | REST API framework with automatic Swagger UI |
| scikit-learn | TF-IDF vectorization + cosine similarity |
| numpy | Matrix operations for similarity scores |
| Pydantic v2 | Input validation + response serialization |
| uvicorn | ASGI server for FastAPI |

### Demo 2 — Schema Design
| Tool | Purpose |
|------|---------|
| PostgreSQL | Primary relational database |
| pgvector | Vector similarity search extension |
| pgcrypto | UUID generation (`gen_random_uuid()`) |

### Production Upgrade Path
| Tool | Purpose |
|------|---------|
| sentence-transformers | `all-MiniLM-L6-v2` for 384-dim semantic embeddings |
| Pinecone / pgvector | Vector store for production similarity search |
| diffusers | DreamBooth-LoRA training and inference |
| InsightFace | Face embedding extraction (ArcFace model) |
| Redis | Caching frequent queries |
| SQS / BullMQ | Async job queue for heavy ML tasks |

### AI Model
| Tool | Purpose |
|------|---------|
| Anthropic Claude (`claude-sonnet-4-6`) | Lead classification with HOT/WARM/COLD scoring |
| Prompt caching | Reduces input token cost ~90% on repeated calls |
| Streaming | Prevents HTTP timeout on long AI responses |

### GitHub Repository
https://github.com/Sudharsan2816/keabuilder-ml-assessment
