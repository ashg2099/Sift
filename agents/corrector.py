from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, validator
from typing import List, Optional
from agents.evidence_hunter import Evidence

try:
    from tavily import TavilyClient
    _tavily_client = (
        TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        if os.environ.get("TAVILY_API_KEY")
        else None
    )
except Exception:
    _tavily_client = None

# ── Output Model ──────────────────────────────────────
class Correction(BaseModel):
    has_correction: bool           # False if no better info found
    correct_info: str              # What is actually true
    original_misattribution: str   # What was wrong about the claim
    explanation: str               # Plain-English user-facing note
    source_url: str                # Best source URL found (empty string if none)
    source_name: str               # Human-readable source name

    # Guard against LLaMA returning "true"/"false" strings
    @validator("has_correction", pre=True)
    def coerce_bool(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() not in ("false", "0", "no", "")

# ── Setup ─────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(Correction, method="json_mode")

# ── Prompt: evidence-based correction ─────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a fact-correction specialist. A claim has been marked as UNCERTAIN or FALSE. Return your response as a JSON object with these exact fields:
{{"has_correction": true, "correct_info": "...", "original_misattribution": "...", "explanation": "...", "source_url": "...", "source_name": "..."}}

Field names MUST be exactly: "has_correction", "correct_info", "original_misattribution", "explanation", "source_url", "source_name".

Your job is to find and explain the CORRECT information.

Given the claim and any fresh evidence retrieved, determine:
1. What is actually true (if you can establish it from the evidence)
2. What was wrong or misleading about the original claim
3. A clear, friendly plain-English explanation for the user

Rules:
- Only assert corrections you can support from the provided evidence
- If the evidence is insufficient to establish what is correct, set has_correction=False
- Be specific — name real sources, real numbers, real organizations
- Keep explanation under 2 sentences, written for a general audience
- If the claim is misattributed (said to come from X but actually from Y), make that clear
- source_url should be the most credible URL from the evidence, empty string if none"""),

    ("human", """Original claim: {claim}

This claim was marked as: {verdict}

Fresh evidence retrieved to find the correct information:
{evidence_text}

Based on this evidence, what is the correct information?
Return a correction if you can establish one from the evidence.""")
])

chain = prompt | structured_llm

# ── Prompt: knowledge-based fallback ──────────────────
knowledge_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a fact-correction specialist. A claim has been marked as UNCERTAIN or FALSE. Return your response as a JSON object with these exact fields:
{{"has_correction": true, "correct_info": "...", "original_misattribution": "...", "explanation": "...", "source_url": "...", "source_name": "..."}}

Field names MUST be exactly: "has_correction", "correct_info", "original_misattribution", "explanation", "source_url", "source_name".

Use your training knowledge to provide the most accurate correction you can.

Think carefully about:
1. MISATTRIBUTION — Is this stat/quote attributed to the wrong org? Who actually published it?
   Example: A "$X billion" figure widely cited by one body but originating from a different research group.
2. WRONG NUMBERS — Is the specific figure incorrect? What is the real number?
3. WRONG FACTS — Is a location, date, person, or event incorrect?
4. CONTEXT NEEDED — Is the claim technically true but missing crucial context?

Rules:
- Be specific — name the real original source, publication year, exact numbers if known
- For misattributed statistics: clearly state "This figure originally came from [X], not [Y]"
- For wrong numbers: state the correct figure and its source
- For uncertain facts: explain what IS known and what remains unverified
- Keep explanation under 2 sentences, written for a general audience
- Set source_url to empty string (no live source)
- Set source_name to "Model knowledge (verify independently)"
- Only set has_correction=True if you are genuinely confident — don't hallucinate"""),

    ("human", """Original claim: {claim}

This claim was marked as: {verdict}

Based on your training knowledge, what is the correct or more accurate information?
Pay special attention to whether this statistic or claim is being attributed to the right source.""")
])

knowledge_chain = knowledge_prompt | structured_llm

# ── Live search specifically for the correction ───────
def _search_for_truth(claim: str) -> List[Evidence]:
    """
    Targeted live search to find what is actually true about a
    claim that failed verification. Tavily first (full web),
    then Guardian + NewsAPI as news supplements.
    """
    evidence = []

    # Tavily — full web search, fetches actual page content
    if _tavily_client:
        try:
            # Search for both the claim and its fact-check specifically
            for query in [claim, f"fact check {claim}"]:
                response = _tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=4,
                    include_answer=False,
                )
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
            print(f"  Corrector Tavily: {len(evidence)} results")
        except Exception as e:
            print(f"  Corrector Tavily error: {e}")

    # Guardian live search
    guardian_key = os.environ.get("GUARDIAN_API_KEY", "")
    if guardian_key:
        try:
            r = requests.get(
                "https://content.guardianapis.com/search",
                params={
                    "q": claim[:120],
                    "api-key": guardian_key,
                    "show-fields": "bodyText,headline",
                    "page-size": 5,
                    "order-by": "relevance",
                },
                timeout=8,
            )
            r.raise_for_status()
            for item in r.json().get("response", {}).get("results", []):
                fields = item.get("fields", {})
                body = fields.get("bodyText", "")
                url = item.get("webUrl", "")
                if body:
                    evidence.append(Evidence(
                        text=body[:600],
                        source=url,
                        relevance_score=0.75,
                    ))
        except Exception as e:
            print(f"  Corrector Guardian error: {e}")

    # NewsAPI live search
    newsapi_key = os.environ.get("NEWSAPI_KEY", "")
    if newsapi_key:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": claim[:120],
                    "apiKey": newsapi_key,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": 4,
                },
                timeout=8,
            )
            r.raise_for_status()
            for article in r.json().get("articles", []):
                title = article.get("title", "") or ""
                description = article.get("description", "") or ""
                content = article.get("content", "") or ""
                url = article.get("url", "")
                text = " ".join(filter(None, [title, description, content])).strip()
                if len(text) > 40:
                    evidence.append(Evidence(
                        text=text[:600],
                        source=url,
                        relevance_score=0.70,
                    ))
        except Exception as e:
            print(f"  Corrector NewsAPI error: {e}")

    return evidence


# ── Main Function ─────────────────────────────────────
def correct_claim(claim: str, verdict: str) -> Optional[Correction]:
    """
    Only runs when verdict is UNCERTAIN or FALSE.
    Searches for what is actually true and returns a correction.
    Phase 1: evidence-based (live Guardian + NewsAPI)
    Phase 2: knowledge fallback (LLM training data) — always attempted
    """
    print(f"\n[CORRECTOR] Searching for correct information...")

    evidence = _search_for_truth(claim)

    try:
        # Phase 1: try evidence-based correction (only if we have evidence)
        if evidence:
            evidence_text = "\n".join(
                f"[{i+1}] {e.source}: {e.text}"
                for i, e in enumerate(evidence)
            )

            correction = chain.invoke({
                "claim": claim,
                "verdict": verdict,
                "evidence_text": evidence_text,
            })

            if correction.has_correction:
                print(f"  ✓ Evidence-based correction: {correction.explanation[:80]}...")
                return correction

            print("  Evidence insufficient → trying knowledge-based correction...")
        else:
            print("  No live evidence found → trying knowledge-based correction...")

        # Phase 2: fallback to model knowledge (always runs if phase 1 fails)
        correction = knowledge_chain.invoke({
            "claim": claim,
            "verdict": verdict,
        })

        if correction.has_correction:
            print(f"  ✓ Knowledge-based correction: {correction.explanation[:80]}...")
        else:
            print("  No correction could be established")

        return correction

    except Exception as e:
        print(f"  Corrector LLM error: {e}")
        return None
