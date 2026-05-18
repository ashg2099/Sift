"""
eval/benchmark.py: RAGAS evaluation harness for Sift
Runs LIAR dataset claims through the pipeline and measures
faithfulness, answer relevancy, and label accuracy.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
from typing import List, Dict
from collections import defaultdict

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

from graph.pipeline import run_sift

# =================== Config ===================
N_SAMPLES   = 20          # claims to evaluate: increase for more thorough eval
OUTPUT_FILE = "eval/results.json"

# =================== Label mapping ===================
# LIAR has 6 labels → map to our 3
LABEL_MAP = {
    "true":        "TRUE",
    "mostly-true": "TRUE",
    "half-true":   "UNCERTAIN",
    "barely-true": "UNCERTAIN",
    "false":       "FALSE",
    "pants-fire":  "FALSE",
}

# =================== Curated test set ===================
# Handcrafted claims with known ground truth — covers all 3 verdict types
# and spans the topics in our ingested evidence base.
TEST_CLAIMS = [
    # TRUE claims
    {"claim": "The US GDP grew 3.2% in Q2 2024.",                                          "ground_truth_label": "TRUE"},
    {"claim": "The unemployment rate fell to 3.7% in October 2024.",                       "ground_truth_label": "TRUE"},
    {"claim": "OpenAI released GPT-4 in March 2023.",                                      "ground_truth_label": "TRUE"},
    {"claim": "Global temperatures in 2023 were approximately 1.45°C above pre-industrial levels.", "ground_truth_label": "TRUE"},
    {"claim": "Nvidia's market capitalisation exceeded $3 trillion in 2024.",               "ground_truth_label": "TRUE"},
    {"claim": "The Federal Reserve raised interest rates multiple times in 2022 and 2023.", "ground_truth_label": "TRUE"},
    {"claim": "Inflation in the United States peaked above 9% in 2022.",                   "ground_truth_label": "TRUE"},
    # FALSE claims
    {"claim": "The US GDP shrank by 5% in Q2 2024.",                                       "ground_truth_label": "FALSE"},
    {"claim": "OpenAI was founded in 2010.",                                                "ground_truth_label": "FALSE"},
    {"claim": "The unemployment rate rose to 8% in 2024.",                                 "ground_truth_label": "FALSE"},
    {"claim": "The Eiffel Tower is located in Berlin, Germany.",                            "ground_truth_label": "FALSE"},
    {"claim": "Nvidia's stock price fell 80% in 2024.",                                    "ground_truth_label": "FALSE"},
    {"claim": "The Federal Reserve cut interest rates to zero in 2024.",                   "ground_truth_label": "FALSE"},
    {"claim": "Global CO2 levels decreased significantly in 2023.",                        "ground_truth_label": "FALSE"},
    # UNCERTAIN claims
    {"claim": "The US economy will enter a recession in 2025.",                            "ground_truth_label": "UNCERTAIN"},
    {"claim": "AI will replace 50% of all jobs within the next five years.",               "ground_truth_label": "UNCERTAIN"},
    {"claim": "The Federal Reserve kept interest rates at exactly 5.25% throughout 2024.", "ground_truth_label": "UNCERTAIN"},
    {"claim": "Climate change will cause sea levels to rise by 2 metres by 2100.",         "ground_truth_label": "UNCERTAIN"},
    {"claim": "The 2024 US election was decided by fewer than 100,000 votes.",             "ground_truth_label": "UNCERTAIN"},
    {"claim": "Renewable energy will account for more than 50% of US power by 2030.",      "ground_truth_label": "UNCERTAIN"},
]

def load_liar_sample(n: int = N_SAMPLES) -> List[Dict]:
    print(f"Loading {n} claims from curated test set...")
    samples = TEST_CLAIMS[:n]
    for s in samples:
        s.setdefault("original_label", s["ground_truth_label"])
        s.setdefault("speaker", "curated")
    print(f"Loaded {len(samples)} claims\n")
    return samples

# =================== Run pipeline on each claim ===================
def run_eval_pipeline(samples: List[Dict]) -> List[Dict]:
    results = []
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['claim'][:70]}...")
        start = time.time()
        try:
            reports = run_sift(sample["claim"])
            elapsed = round(time.time() - start, 2)
            if reports:
                r = reports[0]
                predicted  = r.get("decision", "UNCERTAIN")
                confidence = r.get("final_confidence", r.get("confidence", 0.0))
                reasoning  = r.get("revised_reasoning", r.get("reasoning", ""))
                evidence   = r.get("supporting_evidence", [])
                print(f"  → {predicted} (truth: {sample['ground_truth_label']}) "
                      f"conf:{confidence:.0%} in {elapsed}s")
                results.append({
                    "claim":              sample["claim"],
                    "ground_truth":       sample["ground_truth_label"],
                    "original_label":     sample["original_label"],
                    "predicted":          predicted,
                    "confidence":         confidence,
                    "reasoning":          reasoning,
                    "supporting_evidence": evidence,
                    "elapsed_s":          elapsed,
                })
            else:
                print("  → No report returned")
        except Exception as e:
            print(f"  → Error: {e}")
            results.append({
                "claim":              sample["claim"],
                "ground_truth":       sample["ground_truth_label"],
                "original_label":     sample["original_label"],
                "predicted":          "ERROR",
                "confidence":         0.0,
                "reasoning":          str(e),
                "supporting_evidence": [],
                "elapsed_s":          round(time.time() - start, 2),
            })
        time.sleep(2)   # respect Groq rate limits
    return results

# =================== Accuracy ===================
def compute_accuracy(results: List[Dict]) -> Dict:
    valid = [r for r in results if r["predicted"] != "ERROR"]
    if not valid:
        return {}
    correct = sum(1 for r in valid if r["predicted"] == r["ground_truth"])
    label_correct = defaultdict(int)
    label_total   = defaultdict(int)
    for r in valid:
        label_total[r["ground_truth"]] += 1
        if r["predicted"] == r["ground_truth"]:
            label_correct[r["ground_truth"]] += 1
    return {
        "overall_accuracy": round(correct / len(valid), 3),
        "correct":          correct,
        "total":            len(valid),
        "avg_confidence":   round(sum(r["confidence"] for r in valid) / len(valid), 3),
        "avg_latency_s":    round(sum(r["elapsed_s"]  for r in valid) / len(valid), 2),
        "per_label": {
            label: {
                "accuracy": round(label_correct[label] / label_total[label], 3),
                "correct":  label_correct[label],
                "total":    label_total[label],
            }
            for label in label_total
        },
    }

# =================== RAGAS ===================
def run_ragas(results: List[Dict]) -> Dict:
    """
    LLM-graded quality metrics using RAGAS.
    Faithfulness  — is the reasoning grounded in retrieved evidence?
    Ans Relevancy — is the reasoning relevant to the claim?
    """
    valid = [r for r in results if r["predicted"] != "ERROR" and r["reasoning"]]
    if not valid:
        return {}
    print(f"\nRunning RAGAS on {len(valid)} results...")

    ragas_dataset = Dataset.from_dict({
        "question":     [r["claim"] for r in valid],
        "answer":       [r["reasoning"] for r in valid],
        "contexts":     [r["supporting_evidence"] if r["supporting_evidence"]
                         else ["No evidence retrieved"] for r in valid],
        "ground_truth": [r["ground_truth"] for r in valid],
    })

    try:
        # Build metrics with Groq via OpenAI-compatible client
        from openai import OpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings

        groq_client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        groq_llm = llm_factory("llama-3.3-70b-versatile", client=groq_client)
        hf_emb   = RagasHFEmbeddings(model="BAAI/bge-m3")

        metrics = [
            Faithfulness(llm=groq_llm),
            AnswerRelevancy(llm=groq_llm, embeddings=hf_emb),
        ]
        result = evaluate(ragas_dataset, metrics=metrics)
        return {
            "faithfulness":     round(float(result["faithfulness"]),     3),
            "answer_relevancy": round(float(result["answer_relevancy"]), 3),
        }
    except Exception as e:
        # RAGAS API changes frequently — fall back to manual faithfulness score
        print(f"  RAGAS auto-eval unavailable ({e})")
        print("  Computing manual faithfulness score instead...")
        return _manual_faithfulness(valid)


def _manual_faithfulness(results: List[Dict]) -> Dict:
    """
    Simple proxy faithfulness score:
    fraction of claims where supporting evidence is non-empty
    (i.e. the verdict is grounded in retrieved docs, not LLM memory).
    """
    grounded = sum(1 for r in results if r.get("supporting_evidence"))
    score = round(grounded / len(results), 3)
    print(f"  Manual faithfulness (evidence-grounded rate): {score:.1%}")
    return {"faithfulness_proxy": score, "grounded": grounded, "total": len(results)}

# =================== Main ===================
def main():
    print("=" * 60)
    print("SIFT EVALUATION HARNESS")
    print("=" * 60)

    samples     = load_liar_sample(N_SAMPLES)
    results     = run_eval_pipeline(samples)
    accuracy    = compute_accuracy(results)
    ragas_scores = run_ragas(results)

    report = {"accuracy": accuracy, "ragas": ragas_scores, "per_claim": results}
    os.makedirs("eval", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy : {accuracy.get('overall_accuracy', 0):.1%}")
    print(f"Correct          : {accuracy.get('correct', 0)}/{accuracy.get('total', 0)}")
    print(f"Avg Confidence   : {accuracy.get('avg_confidence', 0):.1%}")
    print(f"Avg Latency      : {accuracy.get('avg_latency_s', 0)}s per claim")
    if ragas_scores and "error" not in ragas_scores:
        print(f"\nRAGAS Faithfulness  : {ragas_scores.get('faithfulness', 0):.3f}")
        print(f"RAGAS Ans Relevancy : {ragas_scores.get('answer_relevancy', 0):.3f}")
    print("\nPer-label breakdown:")
    for label, stats in accuracy.get("per_label", {}).items():
        print(f"  {label:12s}: {stats['accuracy']:.1%}  ({stats['correct']}/{stats['total']})")
    print(f"\nFull report → {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()