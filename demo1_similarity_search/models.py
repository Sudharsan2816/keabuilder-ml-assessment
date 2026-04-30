from pydantic import BaseModel
from typing import Optional


class QueryInput(BaseModel):
    query: str
    top_k: Optional[int] = 3
    corpus_type: Optional[str] = "leads"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "I want to sell my coaching program online",
                    "top_k": 3,
                    "corpus_type": "leads"
                }
            ]
        }
    }


class SimilarEntry(BaseModel):
    id: int
    text: str
    category: str
    similarity_score: float
    extra_fields: dict


class SimilarityResult(BaseModel):
    query: str
    corpus_type: str
    top_match: SimilarEntry
    all_matches: list[SimilarEntry]
    method: str
    corpus_size: int
    production_upgrade: str
