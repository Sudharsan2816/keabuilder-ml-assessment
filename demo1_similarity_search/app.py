from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models import QueryInput, SimilarityResult, SimilarEntry
from data import LEADS_CORPUS, PROMPTS_CORPUS, PRODUCTION_ARCHITECTURE

app = FastAPI(
    title="KeaBuilder ML Similarity Search",
    description="""
    ML-powered similarity search for KeaBuilder.

    **Current:** TF-IDF Cosine Similarity (lightweight, zero-GPU)
    **Production:** sentence-transformers + pgvector (semantic search)

    Supports both leads corpus and prompts corpus.
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CORPUS_MAP = {
    "leads": LEADS_CORPUS,
    "prompts": PROMPTS_CORPUS
}


def run_similarity_search(query: str, corpus: list, top_k: int) -> list[SimilarEntry]:
    texts = [item["text"] for item in corpus]
    all_texts = texts + [query]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=10000
    )

    tfidf_matrix = vectorizer.fit_transform(all_texts)
    query_vector = tfidf_matrix[-1]
    corpus_matrix = tfidf_matrix[:-1]
    scores = cosine_similarity(query_vector, corpus_matrix)[0]

    results = []
    for i, score in enumerate(scores):
        item = corpus[i]
        extra = {k: v for k, v in item.items() if k not in ["id", "text", "category"]}
        results.append(SimilarEntry(
            id=item["id"],
            text=item["text"],
            category=item["category"],
            similarity_score=round(float(score), 4),
            extra_fields=extra
        ))

    results.sort(key=lambda x: x.similarity_score, reverse=True)
    return results[:top_k]


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "KeaBuilder ML Similarity Search",
        "version": "1.0.0",
        "status": "running",
        "available_corpus": list(CORPUS_MAP.keys()),
        "corpus_sizes": {k: len(v) for k, v in CORPUS_MAP.items()},
        "endpoints": {
            "search": "POST /find-similar",
            "corpus": "GET /corpus/{corpus_type}",
            "architecture": "GET /architecture",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "service": "ml-similarity-search",
        "corpus_loaded": {k: len(v) for k, v in CORPUS_MAP.items()}
    }


@app.get("/corpus/{corpus_type}", tags=["Data"])
def get_corpus(corpus_type: str):
    if corpus_type not in CORPUS_MAP:
        raise HTTPException(
            status_code=404,
            detail=f"Corpus '{corpus_type}' not found. Available: {list(CORPUS_MAP.keys())}"
        )
    corpus = CORPUS_MAP[corpus_type]
    return {
        "corpus_type": corpus_type,
        "entries": corpus,
        "count": len(corpus)
    }


@app.get("/architecture", tags=["ML Architecture"])
def get_architecture():
    """Returns current vs production architecture comparison for similarity search."""
    return PRODUCTION_ARCHITECTURE


@app.post("/find-similar", response_model=SimilarityResult, tags=["Similarity Search"])
def find_similar(query_input: QueryInput):
    """
    Find most similar entries to a user query using TF-IDF cosine similarity.

    - **corpus_type**: 'leads' (lead form inputs) or 'prompts' (AI prompt history)
    - **top_k**: Number of matches to return (default 3, must be >= 1)
    - Returns ranked results with similarity scores
    """
    if not query_input.query or len(query_input.query.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 3 characters"
        )

    corpus_type = query_input.corpus_type or "leads"
    if corpus_type not in CORPUS_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"corpus_type must be one of: {list(CORPUS_MAP.keys())}"
        )

    corpus = CORPUS_MAP[corpus_type]

    # FIX BUG-01 / BUG-02: top_k=0 or negative produced an empty results list,
    # causing an IndexError on matches[0] in the return statement.
    top_k = query_input.top_k if query_input.top_k is not None else 3
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")
    top_k = min(top_k, len(corpus))

    try:
        matches = run_similarity_search(query_input.query, corpus, top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    return SimilarityResult(
        query=query_input.query,
        corpus_type=corpus_type,
        top_match=matches[0],
        all_matches=matches,
        method="TF-IDF Cosine Similarity (ngram_range=1-2, stop_words=english)",
        corpus_size=len(corpus),
        production_upgrade=(
            "Upgrade to sentence-transformers/all-MiniLM-L6-v2 + pgvector "
            "for semantic search. See GET /architecture for migration steps."
        )
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)
