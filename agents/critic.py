from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, validator
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

    # Guard against LLaMA returning strings instead of proper types
    @validator("final_confidence", pre=True)
    def coerce_confidence(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.5

    @validator("confidence_changed", pre=True)
    def coerce_bool(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() not in ("false", "0", "no", "")

# ── Setup ─────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(Critique, method="json_mode")

# ── Prompt ────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an adversarial reviewer of fact-check verdicts. Return your response as a JSON object with these exact fields:
{{"original_decision": "...", "final_decision": "TRUE|FALSE|UNCERTAIN", "final_confidence": 0.0, "confidence_changed": false, "issues_found": [], "revised_reasoning": "..."}}

Field names MUST be exactly: "original_decision", "final_decision", "final_confidence", "confidence_changed", "issues_found", "revised_reasoning".

Your job is to find GENUINE weaknesses — not manufacture criticism where none exists.

Ask yourself:
1. Is the decision (TRUE/FALSE/UNCERTAIN) actually correct given the evidence?
   - Watch for over-aggressive FALSE: if the claim is approximately correct or within
     measurement/rounding range (e.g. 1.1°C vs 1.19°C), it should be UNCERTAIN not FALSE
   - Only keep FALSE if the claim is definitively wrong with a clear contradicting fact
2. Is the confidence score reasonable? Only flag it if it is seriously miscalibrated
   (e.g. 95% confidence on a single weak source is too high — but 85% on 5 strong
   sources is fine, do NOT flag it).
3. Does the reasoning contain any claims NOT present in the evidence?
4. Is there important contradicting evidence that was ignored?

Rules:
- If the verdict and confidence are well-supported → issues_found must be EMPTY []
- Only add an issue if it is a real, specific problem — not a generic comment
- NEVER add issues like "confidence score too high" unless confidence is genuinely
  unjustified (e.g. above 80% with only 1 weak source)
- NEVER add issues like "overconfidence in evidence quality" as a blanket statement
- A TRUE verdict with strong multi-source evidence deserves 0 issues
- Only lower confidence if the evidence genuinely does not support the current score
- Quality over quantity — 0 real issues is better than 3 invented ones

CRITICAL — How to write revised_reasoning:
Write like a sourced news article. Name the specific organisations, publications,
or official bodies inline.
Good: "According to the BEA's final revised data, US GDP grew 3.0% in Q2 2024,
       not 3.2%. Reuters and The Guardian reported the same figure."
Bad:  "The evidence shows this claim is false." (too vague — name the source)

Never write "sources indicate" or "evidence suggests" — always say WHO."""),

    ("human", """Original claim: {claim}

Evidence used:
{evidence_text}

Original verdict:
- Decision: {decision}
- Confidence: {confidence}
- Reasoning: {reasoning}
- Supporting sources: {supporting}
- Contradicting sources: {contradicting}

Review this verdict. Only add issues if they are real and specific.
If the verdict is solid, return issues_found as an empty list.
In revised_reasoning, cite source names inline.""")
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