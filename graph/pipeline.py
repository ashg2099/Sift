from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from agents.claim_extractor import extract_claims, Claim
from agents.evidence_hunter import hunt_evidence, Evidence
from agents.synthesis import synthesize_verdict, Verdict
from agents.critic import critique_verdict, Critique
from agents.corrector import correct_claim, Correction

import os

# ── LangFuse Tracing ──────────────────────────────────
def _get_langfuse_handler():
    """Create a fresh LangFuse handler per pipeline run. Reads keys from env vars."""
    try:
        from langfuse.langchain import CallbackHandler as LangfuseCallback
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            handler = LangfuseCallback()   # v3: reads LANGFUSE_* env vars automatically
            print("[Sift] LangFuse tracing enabled")
            return handler
    except Exception as e:
        print(f"[Sift] LangFuse not available: {e}")
    return None

# ── Shared State ──────────────────────────────────────
# This is the "memory" passed between every agent
class SiftState(TypedDict):
    # Input
    original_text: str

    # After claim extractor
    claims: List[dict]
    current_claim_index: int

    # After evidence hunter
    evidence: List[dict]
    evidence_found: bool
    retrieval_attempts: int

    # After synthesis
    verdict: dict

    # After critic
    critique: dict

    # Final output
    final_reports: List[dict]

# ── Node Functions ────────────────────────────────────
# Each node receives the full state and returns updated state

def extract_node(state: SiftState) -> SiftState:
    print("\n[NODE] Claim Extractor running...")
    result = extract_claims(state["original_text"])
    claims = [c.dict() for c in result.claims]
    print(f"  Extracted {len(claims)} claims")
    return {
        **state,
        "claims": claims,
        "current_claim_index": 0,
        "final_reports": []
    }

def evidence_node(state: SiftState) -> SiftState:
    print("\n[NODE] Evidence Hunter running...")
    idx = state["current_claim_index"]
    claim_text = state["claims"][idx]["text"]
    result = hunt_evidence(claim_text)
    return {
        **state,
        "evidence": [e.dict() for e in result.evidence],
        "evidence_found": result.evidence_found,
        "retrieval_attempts": result.retrieval_attempts
    }

def synthesis_node(state: SiftState) -> SiftState:
    print("\n[NODE] Synthesis Agent running...")
    idx = state["current_claim_index"]
    claim_text = state["claims"][idx]["text"]
    evidence = [Evidence(**e) for e in state["evidence"]]
    verdict = synthesize_verdict(claim_text, evidence)
    return {**state, "verdict": verdict.dict()}

def critic_node(state: SiftState) -> SiftState:
    print("\n[NODE] Critic Agent running...")
    verdict = Verdict(**state["verdict"])
    evidence = [Evidence(**e) for e in state["evidence"]]
    critique = critique_verdict(verdict, evidence)

    # Save this claim's final report
    idx = state["current_claim_index"]
    verdict = Verdict(**state["verdict"])

    # ── Run Correction Agent for UNCERTAIN / FALSE verdicts ───────
    correction_data = None
    if critique.final_decision in ("UNCERTAIN", "FALSE"):
        correction = correct_claim(
            claim=state["claims"][idx]["text"],
            verdict=critique.final_decision,
        )
        if correction and correction.has_correction:
            correction_data = {
                "correct_info": correction.correct_info,
                "original_misattribution": correction.original_misattribution,
                "explanation": correction.explanation,
                "source_url": correction.source_url,
                "source_name": correction.source_name,
            }

    report = {
        "claim": state["claims"][idx]["text"],
        "decision": critique.final_decision,
        "final_confidence": critique.final_confidence,
        "revised_reasoning": critique.revised_reasoning,
        "issues_found": critique.issues_found,
        "supporting_evidence": verdict.supporting_evidence,
        "contradicting_evidence": verdict.contradicting_evidence,
        "retrieval_attempts": state.get("retrieval_attempts", 1),
        "correction": correction_data,
    }

    updated_reports = state["final_reports"] + [report]

    return {
        **state,
        "critique": critique.dict(),
        "final_reports": updated_reports,
        "current_claim_index": idx + 1   # move to next claim
    }

# ── Conditional Edges ─────────────────────────────────
# These functions decide WHERE to go next based on state

def check_evidence(state: SiftState) -> str:
    """After evidence hunting — did we find enough?"""
    if state["evidence_found"]:
        return "proceed"       # go to synthesis
    else:
        return "no_evidence"   # skip synthesis, go straight to critic

def check_more_claims(state: SiftState) -> str:
    """After critic — are there more claims to process?"""
    total_claims = len(state["claims"])
    next_index = state["current_claim_index"]

    if next_index < total_claims:
        return "more_claims"   # loop back to evidence hunter
    else:
        return "done"          # all claims processed, end

# ── Build the Graph ───────────────────────────────────
def build_pipeline():
    graph = StateGraph(SiftState)

    # Add all nodes
    graph.add_node("extract", extract_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("critic", critic_node)

    # Entry point
    graph.set_entry_point("extract")

    # Fixed edges
    graph.add_edge("extract", "evidence")
    graph.add_edge("synthesis", "critic")

    # Conditional edge after evidence hunting
    graph.add_conditional_edges(
        "evidence",
        check_evidence,
        {
            "proceed": "synthesis",      # evidence found → synthesize
            "no_evidence": "critic"      # no evidence → skip to critic
        }
    )

    # Conditional edge after critic
    graph.add_conditional_edges(
        "critic",
        check_more_claims,
        {
            "more_claims": "evidence",   # loop back for next claim
            "done": END                  # all done
        }
    )

    return graph.compile()

# ── Run Pipeline ──────────────────────────────────────
pipeline = build_pipeline()

def run_sift(text: str) -> List[dict]:
    print(f"\n{'='*60}")
    print(f"SIFT PIPELINE STARTING")
    print(f"Input: {text[:80]}...")
    print(f"{'='*60}")

    langfuse_handler = _get_langfuse_handler()

    initial_state = SiftState(
        original_text=text,
        claims=[],
        current_claim_index=0,
        evidence=[],
        evidence_found=False,
        retrieval_attempts=0,
        verdict={},
        critique={},
        final_reports=[]
    )

    try:
        config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        final_state = pipeline.invoke(initial_state, config=config)
    finally:
        # Always flush — captures partial traces even on rate limit errors
        if langfuse_handler:
            try:
                langfuse_handler.flush()
                print("[Sift] LangFuse traces flushed")
            except Exception:
                pass

    print(f"\n{'='*60}")
    print(f"SIFT PIPELINE COMPLETE")
    print(f"{'='*60}")

    return final_state["final_reports"]

# ── Test ──────────────────────────────────────────────
if __name__ == "__main__":
    sample_text = """
    The US Federal Reserve raised interest rates by 0.25% in March 2024.
    Inflation hit 3.2% in February according to government data.
    The Eiffel Tower is located in Berlin, Germany.
    The stock market fell 1.4% following the Fed announcement.
    """

    reports = run_sift(sample_text)

    print("\n\nFINAL VERDICTS:")
    print("="*60)
    for r in reports:
        print(f"\nClaim:      {r['claim']}")
        print(f"Decision:   {r['decision']}")
        print(f"Confidence: {r['confidence']:.0%}")
        print(f"Reasoning:  {r['reasoning']}")
        if r['issues']:
            print(f"Issues:     {r['issues']}")