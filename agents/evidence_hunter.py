from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_postgres import PGVector
from pydantic import BaseModel
from typing import List

# ── Tavily client (primary live retrieval) ────────────────────
try:
    from tavily import TavilyClient
    _tavily_client = (
        TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        if os.environ.get("TAVILY_API_KEY")
        else None
    )
except Exception:
    _tavily_client = None

#    Models
class Evidence(BaseModel):
    text: str
    source: str
    relevance_score: float

class EvidenceResult(BaseModel):
    claim: str
    evidence: List[Evidence]
    evidence_found: bool
    retrieval_attempts: int
    live_retrieval_used: bool = False

#    Setup
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ── Embedder: HF Inference API in prod, local model in dev ───
def _build_embedder():
    hf_token = os.environ.get("HF_TOKEN", "")
    try:
        # Try HF Inference API first (lightweight — no torch needed)
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        embedder = HuggingFaceEndpointEmbeddings(
            model="https://api-inference.huggingface.co/models/BAAI/bge-m3",
            huggingfacehub_api_token=hf_token,
        )
        print("[Sift] Using HuggingFace Inference API for embeddings")
        return embedder
    except Exception:
        pass
    try:
        # Fall back to local model (dev environment with torch installed)
        from langchain_huggingface import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[Sift] Using local HuggingFace model for embeddings")
        return embedder
    except Exception as e:
        print(f"[Sift] Embedder unavailable: {e} — pgvector disabled")
        return None

embedder = _build_embedder()

vector_store = PGVector(
    embeddings=embedder,
    collection_name="veridai_evidence",
    connection=os.environ.get("DATABASE_URL", ""),
) if embedder else None

#    HyDE Retrieval
def generate_hypothetical_doc(claim: str) -> str:
    prompt = f"""Write a short factual news passage that would serve
as evidence for or against this claim: "{claim}"

Write only the passage itself, 2-3 sentences, in journalist style.
Do not mention the claim directly."""
    return llm.invoke(prompt).content

def retrieve_evidence(claim: str, k: int = 5, attempt: int = 1) -> List[Document]:
    if vector_store is None:
        return []
    if attempt == 1:
        hypothetical_doc = generate_hypothetical_doc(claim)
        print(f"  HyDE doc: {hypothetical_doc[:80]}...")
        results = vector_store.similarity_search_with_score(hypothetical_doc, k=k)
    else:
        print(f"  Attempt {attempt}: searching with original claim...")
        results = vector_store.similarity_search_with_score(claim, k=k)
    return results

# ── Tavily: full web search (PRIMARY live source) ─────────────
def _live_tavily(claim: str, k: int = 5) -> List[Evidence]:
    """
    Full web search via Tavily — fetches and extracts content from
    real web pages. Catches research reports, Wikipedia, fact-check
    sites, academic pages that news APIs don't index.
    Free tier: 1,000 searches/month.
    """
    if not _tavily_client:
        return []
    try:
        response = _tavily_client.search(
            query=claim,
            search_depth="advanced",   # fetches full page content
            max_results=k,
            include_answer=False,
        )
        evidence = []
        for result in response.get("results", []):
            content = result.get("content", "") or ""
            url     = result.get("url", "")
            score   = float(result.get("score", 0.8))
            if len(content) > 40:
                evidence.append(Evidence(
                    text=content[:600],
                    source=url,
                    relevance_score=round(score, 3),
                ))
        print(f"  ✓ Tavily: {len(evidence)} results")
        return evidence
    except Exception as e:
        print(f"  Tavily error: {e}")
        return []

# ── Guardian: live news (secondary) ──────────────────────────
def _live_guardian(claim: str, k: int = 4) -> List[Evidence]:
    """Query Guardian API live — good for current events."""
    key = os.environ.get("GUARDIAN_API_KEY", "")
    if not key:
        return []
    try:
        r = requests.get(
            "https://content.guardianapis.com/search",
            params={
                "q": claim[:120],
                "api-key": key,
                "show-fields": "bodyText,headline",
                "page-size": k * 2,
            },
            timeout=8,
        )
        r.raise_for_status()
        results = r.json().get("response", {}).get("results", [])
        evidence = []
        for item in results[:k]:
            fields = item.get("fields", {})
            body = fields.get("bodyText", "")
            url  = item.get("webUrl", "")
            if body:
                evidence.append(Evidence(
                    text=body[:500],
                    source=url,
                    relevance_score=0.72,
                ))
        return evidence
    except Exception as e:
        print(f"  Live Guardian error: {e}")
        return []

# ── NewsAPI: live news (tertiary) ─────────────────────────────
def _live_newsapi(claim: str, k: int = 3) -> List[Evidence]:
    """Query NewsAPI live — 100 req/day on free tier."""
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return []
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": claim[:120],
                "apiKey": key,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": k * 2,
            },
            timeout=8,
        )
        r.raise_for_status()
        articles = r.json().get("articles", [])
        evidence = []
        for article in articles[:k]:
            title       = article.get("title", "") or ""
            description = article.get("description", "") or ""
            content     = article.get("content", "") or ""
            url         = article.get("url", "")
            text = " ".join(filter(None, [title, description, content])).strip()
            if len(text) > 40:
                evidence.append(Evidence(
                    text=text[:500],
                    source=url,
                    relevance_score=0.68,
                ))
        return evidence
    except Exception as e:
        print(f"  Live NewsAPI error: {e}")
        return []

#    Main Function
def hunt_evidence(claim: str, max_attempts: int = 2) -> EvidenceResult:
    print(f"\nHunting evidence for: {claim}")

    all_evidence = []
    attempt = 1
    live_used = False
    best_score = 1.0

    # ── Phase 1: pgvector (HyDE then direct) ──────────────────────
    while attempt <= max_attempts:
        results = retrieve_evidence(claim, k=5, attempt=attempt)

        if results:
            best_score = min(score for _, score in results)

        relevant = [(doc, score) for doc, score in results if score < 0.6]

        if relevant:
            all_evidence = [
                Evidence(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=round(1 - score, 3),
                )
                for doc, score in relevant
            ]
            print(f"  ✓ pgvector: {len(all_evidence)} chunks (best score: {best_score:.3f})")
            break
        else:
            print(f"  Attempt {attempt}: no relevant evidence in pgvector, retrying...")
            attempt += 1

    # ── Phase 2: Tavily web search (always runs — primary live source) ──
    # Tavily fetches full page content from the open web: research reports,
    # Wikipedia, fact-check sites, PDFs — everything news APIs miss.
    print("  Running Tavily web search...")
    tavily_evidence = _live_tavily(claim, k=5)

    if tavily_evidence:
        all_evidence = all_evidence + tavily_evidence
        live_used = True

    # ── Phase 3: Guardian + NewsAPI (supplement for breaking news) ──
    print("  Supplementing with news APIs...")
    news_evidence  = _live_guardian(claim, k=3)
    news_evidence += _live_newsapi(claim, k=2)

    if news_evidence:
        all_evidence = all_evidence + news_evidence
        live_used = True

    if all_evidence:
        print(f"  ✓ Total evidence: {len(all_evidence)} pieces (pgvector + tavily + news)")
    else:
        print("  ✗ No evidence found from any source")

    return EvidenceResult(
        claim=claim,
        evidence=all_evidence,
        evidence_found=len(all_evidence) > 0,
        retrieval_attempts=attempt,
        live_retrieval_used=live_used,
    )

#    Test
if __name__ == "__main__":
    test_claims = [
        "The World Economic Forum estimated misinformation costs $78 billion annually",
        "The US GDP grew 3.2% in Q2 2024",
        "The Eiffel Tower is located in Berlin",
    ]

    for claim in test_claims:
        result = hunt_evidence(claim)
        print(f"\nClaim: {claim}")
        print(f"Evidence found: {result.evidence_found}")
        print(f"Attempts needed: {result.retrieval_attempts}")
        for ev in result.evidence:
            print(f"  [{ev.relevance_score}] {ev.source}: {ev.text[:80]}...")
        print("-" * 60)
