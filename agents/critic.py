from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Literal
from agents.synthesis import Verdict, synthesize_verdict
from agents.evidence_hunter import Evidence, hunt_evidence

# ── Output Model ──────────────────────────────────────
class Critique(BaseModel):
    original_decision: str
    final_decision: Literal["TRUE", "FALSE", "UNCERTAIN"]
    final_confidence: float
    confidence_changed: bool
    issues_found: List[str]      # problems with the original verdict
    revised_reasoning: str       # updated explanation after critique

# ── Setup ─────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(Critique)

# ── Prompt ────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an adversarial reviewer of fact-check verdicts.
Your job is to challenge the verdict and find any weaknesses.

Ask yourself:
1. Is every part of the verdict actually supported by the evidence?
2. Is the confidence score too high given the evidence quality?
3. Are there alternative interpretations the evidence allows?
4. Is anything stated in the reasoning NOT present in the evidence?

Rules:
- If the verdict is well-supported, keep it but confirm it
- If confidence is too high for thin evidence, lower it
- If reasoning contains unsupported claims, flag them
- If verdict should change based on your review, change it
- Be strict — overconfidence is worse than underconfidence"""),

    ("human", """Original claim: {claim}

Evidence used:
{evidence_text}

Original verdict:
- Decision: {decision}
- Confidence: {confidence}
- Reasoning: {reasoning}
- Supporting sources: {supporting}
- Contradicting sources: {contradicting}

Review this verdict critically and return your assessment.""")
])

chain = prompt | structured_llm

# ── Helper ────────────────────────────────────────────
def format_evidence(evidence: List[Evidence]) -> str:
    if not evidence:
        return "No evidence was retrieved."
    return "\n".join(
        f"[{i+1}] {ev.source} (score: {ev.relevance_score}): {ev.text}"
        for i, ev in enumerate(evidence)
    )

# ── Main Function ─────────────────────────────────────
def critique_verdict(verdict: Verdict, evidence: List[Evidence]) -> Critique:
    print(f"\nCritiquing verdict: {verdict.decision} ({verdict.confidence:.0%})")

    # No point critiquing if there was no evidence
    if not evidence:
        return Critique(
            original_decision=verdict.decision,
            final_decision="UNCERTAIN",
            final_confidence=0.0,
            confidence_changed=False,
            issues_found=["No evidence was retrieved — cannot verify claim"],
            revised_reasoning="Verdict stands as UNCERTAIN due to lack of evidence."
        )

    critique = chain.invoke({
        "claim": verdict.claim,
        "evidence_text": format_evidence(evidence),
        "decision": verdict.decision,
        "confidence": verdict.confidence,
        "reasoning": verdict.reasoning,
        "supporting": verdict.supporting_evidence,
        "contradicting": verdict.contradicting_evidence
    })

    # Show if the critic changed anything
    if critique.confidence_changed:
        print(f"  Critic adjusted confidence: {verdict.confidence:.0%} → {critique.final_confidence:.0%}")
    if critique.final_decision != verdict.decision:
        print(f"  Critic changed decision: {verdict.decision} → {critique.final_decision}")
    if not critique.issues_found:
        print("  Critic: verdict is solid, no issues found")

    return critique

# ── Test ──────────────────────────────────────────────
if __name__ == "__main__":
    test_claims = [
        "The US GDP grew 3.2% in Q2 2024",
        "The Eiffel Tower is located in Berlin",
    ]

    for claim in test_claims:
        print(f"\n{'='*60}")
        print(f"CLAIM: {claim}")

        # Full pipeline: hunt → synthesize → critique
        evidence_result = hunt_evidence(claim)
        verdict = synthesize_verdict(claim, evidence_result.evidence)
        critique = critique_verdict(verdict, evidence_result.evidence)

        print(f"\nFINAL RESULT:")
        print(f"  Decision:   {critique.final_decision}")
        print(f"  Confidence: {critique.final_confidence:.0%}")
        print(f"  Issues:     {critique.issues_found}")
        print(f"  Reasoning:  {critique.revised_reasoning}")
        print(f"{'='*60}")