from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, validator
from typing import List, Literal
from agents.evidence_hunter import Evidence, hunt_evidence

# ── Output Model ──────────────────────────────────────
class Verdict(BaseModel):
    claim: str
    decision: Literal["TRUE", "FALSE", "UNCERTAIN"]
    confidence: float        # 0.0 to 1.0
    reasoning: str           # why this decision was made
    supporting_evidence: List[str]   # which sources support it
    contradicting_evidence: List[str]  # which sources contradict it

    # Guard against LLaMA returning confidence as a string
    @validator("confidence", pre=True)
    def coerce_confidence(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.5

# ── Setup ─────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(Verdict)

# ── Prompt ────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional fact-checker. Your job is to
evaluate a claim against provided evidence and return a verdict.

Rules:
- TRUE: evidence clearly supports the claim
- FALSE: evidence clearly contradicts the claim
- UNCERTAIN: evidence is missing, weak, or conflicting

Be conservative — if evidence is thin, say UNCERTAIN.
Never make up facts not present in the evidence.
Confidence must reflect how strongly the evidence supports your decision.
0.9+ means very strong evidence, 0.5 means borderline.

CRITICAL — How to write the reasoning field:
Write the reasoning like a journalist citing sources inline. Always name the
specific organisation, publication, or report that provided the data.
Examples of good reasoning:
  "According to the U.S. Bureau of Economic Analysis (BEA), GDP grew 3.0% in Q2 2024,
   contradicting the claimed 3.2%. The Guardian reported the same figure."
  "The Federal Reserve's official statement, cited by Reuters, confirms rates are at
   3.5–3.75%, not 5.25% as claimed."
  "The World Health Organization (WHO) reported in its 2024 bulletin that..."

Never write vague reasoning like "evidence supports this" — always say WHO says so."""),

    ("human", """Claim to verify: {claim}

Evidence retrieved:
{evidence_text}

Return your verdict. In the reasoning field, cite specific source names inline
(e.g. "According to Reuters...", "The BEA reported...", "Per The Guardian...").
Make the reasoning read like a sourced fact-check, not a generic summary.""")
])

chain = prompt | structured_llm

# ── Helper ────────────────────────────────────────────
def format_evidence(evidence: List[Evidence]) -> str:
    if not evidence:
        return "No evidence found."
    
    formatted = []
    for i, ev in enumerate(evidence, 1):
        formatted.append(
            f"[{i}] Source: {ev.source}\n"
            f"    Relevance: {ev.relevance_score}\n"
            f"    Text: {ev.text}"
        )
    return "\n\n".join(formatted)

# ── Main Function ─────────────────────────────────────
def synthesize_verdict(claim: str, evidence: List[Evidence]) -> Verdict:
    print(f"\nSynthesizing verdict for: {claim}")
    
    # Handle case where no evidence was found at all
    if not evidence:
        print("  No evidence — returning UNCERTAIN")
        return Verdict(
            claim=claim,
            decision="UNCERTAIN",
            confidence=0.0,
            reasoning="No evidence was found in the knowledge base to verify this claim.",
            supporting_evidence=[],
            contradicting_evidence=[]
        )
    
    evidence_text = format_evidence(evidence)
    
    verdict = chain.invoke({
        "claim": claim,
        "evidence_text": evidence_text
    })
    
    print(f"  Decision: {verdict.decision} (confidence: {verdict.confidence})")
    return verdict

# ── Test ──────────────────────────────────────────────
if __name__ == "__main__":
    # Full pipeline test — hunter feeds into synthesis
    test_claims = [
        "The US GDP grew 3.2% in Q2 2024",
        "The Eiffel Tower is located in Berlin",
        "Unemployment fell to 3.7% in October 2024",
    ]

    for claim in test_claims:
        # Step 1 — hunt for evidence
        evidence_result = hunt_evidence(claim)
        
        # Step 2 — synthesize verdict from that evidence
        verdict = synthesize_verdict(claim, evidence_result.evidence)
        
        print(f"\n{'='*60}")
        print(f"CLAIM:      {verdict.claim}")
        print(f"VERDICT:    {verdict.decision}")
        print(f"CONFIDENCE: {verdict.confidence:.0%}")
        print(f"REASONING:  {verdict.reasoning}")
        print(f"SUPPORTING: {verdict.supporting_evidence}")
        print(f"CONTRADICTING: {verdict.contradicting_evidence}")
        print(f"{'='*60}")