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

    @validator("confidence", always=True)
    def fix_zero_confidence(cls, v, values):
        """
        LLaMA sometimes outputs 0.0 for UNCERTAIN verdicts, which is wrong —
        0.0 means "definitely false", not "I don't know".
        UNCERTAIN with no evidence = 0.3, UNCERTAIN with some evidence = 0.4.
        Only touch confidence if it's exactly 0.0 AND decision is UNCERTAIN.
        TRUE/FALSE verdicts are never changed.
        """
        decision = values.get("decision", "")
        if v == 0.0 and decision == "UNCERTAIN":
            return 0.4
        return v

# ── Setup ─────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(Verdict, method="json_mode")

# ── Prompt ────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional fact-checker. Your job is to
evaluate a claim against provided evidence and return a verdict as a JSON object with these exact fields:
{{"claim": "...", "decision": "TRUE|FALSE|UNCERTAIN", "confidence": 0.0, "reasoning": "...", "supporting_evidence": [], "contradicting_evidence": []}}

Field names MUST be exactly: "claim", "decision", "confidence", "reasoning", "supporting_evidence", "contradicting_evidence".

Rules:
- TRUE: evidence clearly supports the claim
- FALSE: evidence clearly and directly contradicts the claim with a different fact
- UNCERTAIN: evidence is missing, weak, conflicting, or the numbers are close enough to be within measurement uncertainty

Important nuances:
- Do NOT mark FALSE just because a number is slightly different (e.g. 1.1°C vs 1.19°C — that is rounding, not a false claim)
- Do NOT mark FALSE if the claim is approximately correct or a reasonable approximation
- Only mark FALSE if the claim is definitively wrong (e.g. wrong location, wrong person, clearly fabricated statistic)
- When in doubt between FALSE and UNCERTAIN, choose UNCERTAIN

Be conservative — if evidence is thin or the difference is minor, say UNCERTAIN.
Never make up facts not present in the evidence.
Confidence must reflect how strongly the evidence supports your decision:
- TRUE/FALSE with strong multi-source evidence: 0.80–0.95
- TRUE/FALSE with weak or single-source evidence: 0.55–0.75
- UNCERTAIN with some relevant evidence: 0.35–0.50
- UNCERTAIN with no useful evidence: 0.25–0.35
NEVER set confidence to 0.0 — that is not a valid output.

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
            confidence=0.3,
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