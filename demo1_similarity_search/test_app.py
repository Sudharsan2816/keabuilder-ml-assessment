"""
Test suite for keabuilder-ml-assessment / demo1_similarity_search.

Bugs surfaced by this suite
──────────────────────────
BUG-01  top_k=0  → min(0, len)=0 → empty matches list → matches[0] IndexError  (TC-20)
BUG-02  top_k<0  → results[:-n] may be empty → same IndexError path            (TC-21)

Fix: validate top_k >= 1 before the min() call in find_similar().
"""
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# ── Root / Health ──────────────────────────────────────────────────────────────
class TestRootAndHealth:
    def test_root_returns_service_info(self):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "KeaBuilder ML Similarity Search"
        assert "leads" in body["available_corpus"]
        assert "prompts" in body["available_corpus"]

    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["corpus_loaded"]["leads"] == 10
        assert body["corpus_loaded"]["prompts"] == 5


# ── GET /corpus ────────────────────────────────────────────────────────────────
class TestCorpusEndpoint:
    def test_leads_returns_10_entries(self):
        r = client.get("/corpus/leads")
        assert r.status_code == 200
        body = r.json()
        assert body["corpus_type"] == "leads"
        assert body["count"] == 10
        assert len(body["entries"]) == 10

    def test_prompts_returns_5_entries(self):
        r = client.get("/corpus/prompts")
        assert r.status_code == 200
        assert r.json()["count"] == 5

    def test_unknown_corpus_returns_404(self):
        r = client.get("/corpus/unknown_type")
        assert r.status_code == 404

    def test_each_lead_entry_has_valid_lead_score(self):
        r = client.get("/corpus/leads")
        assert r.status_code == 200
        for entry in r.json()["entries"]:
            assert "lead_score" in entry
            assert entry["lead_score"] in ("HOT", "WARM", "COLD")


# ── POST /find-similar – Happy Path ───────────────────────────────────────────
class TestFindSimilarHappyPath:
    def test_tc01_coaching_query_top_match(self):
        """TC-01  coaching query → top match id 2 (coaching) or 5 (webinar funnel)
        Expected per sample_output.json: id=2, score ≈ 0.3891"""
        r = client.post("/find-similar", json={
            "query": "I want to sell my coaching program online",
            "top_k": 3,
            "corpus_type": "leads",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["corpus_type"] == "leads"
        assert len(body["all_matches"]) == 3
        assert body["top_match"]["id"] in (2, 5), (
            f"Expected id 2 or 5, got {body['top_match']['id']}"
        )

    def test_tc02_email_automation_top_match(self):
        """TC-02  email automation → top match id=3 (automation)
        Expected per sample_output.json: score ≈ 0.4756"""
        r = client.post("/find-similar", json={
            "query": "automate email follow-ups for my leads",
            "top_k": 2,
            "corpus_type": "leads",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["top_match"]["id"] == 3
        assert body["top_match"]["category"] == "automation"
        assert body["top_match"]["similarity_score"] > 0.40
        assert len(body["all_matches"]) == 2

    def test_tc03_prompts_corpus_ad_copy(self):
        """TC-03  ad copy query in prompts corpus → top match id=101, category=ad_copy
        Expected per sample_output.json: score ≈ 0.5823"""
        r = client.post("/find-similar", json={
            "query": "write ad copy for fitness coaching",
            "top_k": 2,
            "corpus_type": "prompts",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["top_match"]["id"] == 101
        assert body["top_match"]["category"] == "ad_copy"
        assert body["top_match"]["similarity_score"] > 0.50

    def test_tc04_all_matches_sorted_descending(self):
        """TC-04  all_matches must be ordered by similarity_score DESC"""
        r = client.post("/find-similar", json={
            "query": "funnel landing page conversion optimisation",
            "top_k": 5,
            "corpus_type": "leads",
        })
        assert r.status_code == 200
        scores = [m["similarity_score"] for m in r.json()["all_matches"]]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not sorted descending: {scores}"
        )

    def test_tc05_top_match_equals_all_matches_first(self):
        """TC-05  top_match must be identical to all_matches[0]"""
        r = client.post("/find-similar", json={
            "query": "CRM manage all my leads in one dashboard",
            "top_k": 3,
            "corpus_type": "leads",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["top_match"]["id"] == body["all_matches"][0]["id"]
        assert body["top_match"]["similarity_score"] == body["all_matches"][0]["similarity_score"]

    def test_tc06_method_field_mentions_tfidf(self):
        """TC-06  method field must contain 'TF-IDF'"""
        r = client.post("/find-similar", json={
            "query": "landing page A/B testing",
            "corpus_type": "leads",
        })
        assert r.status_code == 200
        assert "TF-IDF" in r.json()["method"]

    def test_tc07_corpus_size_correct_for_leads(self):
        """TC-07  corpus_size in response must be 10 for leads"""
        r = client.post("/find-similar", json={"query": "build funnels", "corpus_type": "leads"})
        assert r.status_code == 200
        assert r.json()["corpus_size"] == 10

    def test_tc08_corpus_size_correct_for_prompts(self):
        """TC-08  corpus_size must be 5 for prompts"""
        r = client.post("/find-similar", json={"query": "write email copy", "corpus_type": "prompts"})
        assert r.status_code == 200
        assert r.json()["corpus_size"] == 5

    def test_tc09_leads_extra_fields_contain_lead_score(self):
        """TC-09  leads corpus top_match must expose lead_score in extra_fields"""
        r = client.post("/find-similar", json={
            "query": "sales funnel e-commerce store",
            "top_k": 1,
            "corpus_type": "leads",
        })
        assert r.status_code == 200
        extra = r.json()["top_match"]["extra_fields"]
        assert "lead_score" in extra
        assert extra["lead_score"] in ("HOT", "WARM", "COLD")

    def test_tc10_default_corpus_type_is_leads(self):
        """TC-10  omitting corpus_type must default to 'leads'"""
        r = client.post("/find-similar", json={"query": "coaching business leads"})
        assert r.status_code == 200
        assert r.json()["corpus_type"] == "leads"

    def test_tc11_default_top_k_returns_3(self):
        """TC-11  omitting top_k must return exactly 3 results"""
        r = client.post("/find-similar", json={"query": "sales funnel automation"})
        assert r.status_code == 200
        assert len(r.json()["all_matches"]) == 3

    def test_tc12_exact_corpus_text_score_above_90(self):
        """TC-12  verbatim corpus text query → score > 0.90"""
        r = client.post("/find-similar", json={
            "query": "I want to build a sales funnel for my e-commerce store",
            "top_k": 1,
            "corpus_type": "leads",
        })
        assert r.status_code == 200
        score = r.json()["top_match"]["similarity_score"]
        assert score > 0.90, f"Expected > 0.90 for exact match, got {score}"

    def test_tc13_unrelated_query_yields_low_scores(self):
        """TC-13  completely unrelated query → top score < 0.30"""
        r = client.post("/find-similar", json={
            "query": "quantum entanglement photon supercollider",
            "top_k": 1,
            "corpus_type": "leads",
        })
        assert r.status_code == 200
        score = r.json()["top_match"]["similarity_score"]
        assert score < 0.30, f"Expected low score for unrelated query, got {score}"

    def test_tc14_all_scores_in_valid_range(self):
        """TC-14  every similarity_score must be in [0.0, 1.0]"""
        r = client.post("/find-similar", json={
            "query": "funnels landing pages conversion email automation",
            "top_k": 10,
            "corpus_type": "leads",
        })
        assert r.status_code == 200
        for m in r.json()["all_matches"]:
            assert 0.0 <= m["similarity_score"] <= 1.0, (
                f"Score out of range: {m['similarity_score']}"
            )

    def test_tc15_top_k_capped_at_corpus_size(self):
        """TC-15  top_k > corpus size → results capped at 10, no error"""
        r = client.post("/find-similar", json={
            "query": "funnel automation business online",
            "top_k": 999,
            "corpus_type": "leads",
        })
        assert r.status_code == 200
        assert len(r.json()["all_matches"]) == 10


# ── POST /find-similar – Validation & Bug Regression ──────────────────────────
class TestFindSimilarValidation:
    def test_tc16_two_char_query_returns_400(self):
        """TC-16  query of 2 chars → 400"""
        r = client.post("/find-similar", json={"query": "hi", "corpus_type": "leads"})
        assert r.status_code == 400

    def test_tc17_empty_query_returns_400(self):
        """TC-17  empty query string → 400"""
        r = client.post("/find-similar", json={"query": "", "corpus_type": "leads"})
        assert r.status_code == 400

    def test_tc18_whitespace_only_query_returns_400(self):
        """TC-18  whitespace-only query → 400"""
        r = client.post("/find-similar", json={"query": "   ", "corpus_type": "leads"})
        assert r.status_code == 400

    def test_tc19_invalid_corpus_type_returns_400(self):
        """TC-19  unknown corpus_type in POST body → 400"""
        r = client.post("/find-similar", json={
            "query": "coaching program online",
            "corpus_type": "does_not_exist",
        })
        assert r.status_code == 400

    def test_tc20_bug01_top_k_zero_must_be_rejected(self):
        """
        TC-20  BUG-01 — top_k=0:
          Before fix: min(0, 10)=0 → run_similarity_search returns [] → matches[0] IndexError → HTTP 500
          After fix:  validated before min() → HTTP 400
        """
        r = client.post("/find-similar", json={
            "query": "sales funnel email automation",
            "top_k": 0,
            "corpus_type": "leads",
        })
        assert r.status_code in (400, 422), (
            f"BUG-01 NOT FIXED — top_k=0 was not rejected; "
            f"got HTTP {r.status_code}: {r.text}"
        )

    def test_tc21_bug02_negative_top_k_must_be_rejected(self):
        """
        TC-21  BUG-02 — top_k=-1:
          results[:-1] slices off the last item (returns 9 items) so matches[0] exists for -1,
          but for top_k <= -10 the slice returns [] → IndexError.
          Regardless, negative values have undefined intent and must be rejected → HTTP 400.
        """
        r = client.post("/find-similar", json={
            "query": "coaching program funnel",
            "top_k": -1,
            "corpus_type": "leads",
        })
        assert r.status_code in (400, 422), (
            f"BUG-02 NOT FIXED — negative top_k was accepted; "
            f"got HTTP {r.status_code}: {r.text}"
        )
