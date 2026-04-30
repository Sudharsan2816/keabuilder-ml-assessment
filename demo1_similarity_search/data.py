LEADS_CORPUS = [
    {
        "id": 1,
        "text": "I want to build a sales funnel for my e-commerce store",
        "category": "funnel",
        "lead_score": "WARM"
    },
    {
        "id": 2,
        "text": "Looking for lead capture tools for my coaching business",
        "category": "lead_capture",
        "lead_score": "WARM"
    },
    {
        "id": 3,
        "text": "Need automation for follow-up emails after form submission",
        "category": "automation",
        "lead_score": "HOT"
    },
    {
        "id": 4,
        "text": "Want to create landing pages that convert visitors to leads",
        "category": "landing_page",
        "lead_score": "WARM"
    },
    {
        "id": 5,
        "text": "Building a webinar funnel to sell my online course",
        "category": "funnel",
        "lead_score": "HOT"
    },
    {
        "id": 6,
        "text": "How do I integrate Stripe payment into my funnel",
        "category": "payment",
        "lead_score": "HOT"
    },
    {
        "id": 7,
        "text": "Looking for CRM to manage all my leads in one dashboard",
        "category": "crm",
        "lead_score": "WARM"
    },
    {
        "id": 8,
        "text": "Need to automate WhatsApp follow-up messages for my leads",
        "category": "automation",
        "lead_score": "HOT"
    },
    {
        "id": 9,
        "text": "How to A/B test landing page headlines and CTAs",
        "category": "optimization",
        "lead_score": "WARM"
    },
    {
        "id": 10,
        "text": "Track conversion rates and funnel drop-off analytics",
        "category": "analytics",
        "lead_score": "WARM"
    },
]

PROMPTS_CORPUS = [
    {
        "id": 101,
        "text": "Write a compelling Facebook ad for my online fitness coaching",
        "category": "ad_copy",
        "type": "prompt"
    },
    {
        "id": 102,
        "text": "Generate email subject lines for a Black Friday sale",
        "category": "email",
        "type": "prompt"
    },
    {
        "id": 103,
        "text": "Create a landing page headline for a weight loss program",
        "category": "landing_page",
        "type": "prompt"
    },
    {
        "id": 104,
        "text": "Write a thank you email for new lead sign-ups",
        "category": "email",
        "type": "prompt"
    },
    {
        "id": 105,
        "text": "Generate 5 Instagram captions for a productivity app",
        "category": "social_media",
        "type": "prompt"
    },
]

PRODUCTION_ARCHITECTURE = {
    "current": {
        "method": "TF-IDF Cosine Similarity",
        "library": "scikit-learn",
        "storage": "In-memory Python list",
        "strengths": "Zero dependencies, instant startup, interpretable",
        "weaknesses": "Keyword matching only, no semantic understanding"
    },
    "production": {
        "method": "Dense Vector Similarity",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_store": "pgvector (PostgreSQL) or Pinecone",
        "strengths": "Semantic understanding, handles synonyms, multilingual",
        "migration_steps": [
            "1. Install: pip install sentence-transformers",
            "2. Generate embeddings: model.encode(texts) → 384-dim vectors",
            "3. Store in pgvector: CREATE EXTENSION vector; embedding vector(384)",
            "4. Query: SELECT id, 1 - (embedding <=> query_vec) AS score FROM table ORDER BY score DESC LIMIT 5",
            "5. Set threshold: score > 0.7 = strong match"
        ]
    },
    "face_similarity": {
        "method": "Deep Face Embedding Similarity",
        "library": "InsightFace or DeepFace",
        "storage": "pgvector with 512-dim face embeddings",
        "pipeline": [
            "1. Upload image → extract face region",
            "2. Generate 512-dim embedding via InsightFace ArcFace model",
            "3. Store embedding in pgvector",
            "4. Query: cosine similarity search → threshold > 0.7 = same person"
        ]
    }
}
