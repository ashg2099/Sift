from dotenv import load_dotenv
load_dotenv()

import os
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from pydantic import BaseModel
from typing import List

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

#    Setup 
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_store = PGVector(
    embeddings=embedder,
    collection_name="veridai_evidence",
    connection=os.environ["DATABASE_URL"],
)

#    HyDE Retrieval                                     
def generate_hypothetical_doc(claim: str) -> str:
    prompt = f"""Write a short factual news passage that would serve 
as evidence for or against this claim: "{claim}"

Write only the passage itself, 2-3 sentences, in journalist style.
Do not mention the claim directly."""
    
    return llm.invoke(prompt).content

def retrieve_evidence(claim: str, k: int = 5, attempt: int = 1) -> List[Document]:
    if attempt == 1:
        # First attempt, use HyDE
        hypothetical_doc = generate_hypothetical_doc(claim)
        print(f"  HyDE doc: {hypothetical_doc[:80]}...")
        results = vector_store.similarity_search_with_score(hypothetical_doc, k=k)
    else:
        # Retry attempt, search with original claim directly
        print(f"  Attempt {attempt}: searching with original claim...")
        results = vector_store.similarity_search_with_score(claim, k=k)
    
    return results

#    Main Function                                     
def hunt_evidence(claim: str, max_attempts: int = 2) -> EvidenceResult:
    print(f"\nHunting evidence for: {claim}")
    
    all_evidence = []
    attempt = 1
    
    while attempt <= max_attempts:
        results = retrieve_evidence(claim, k=5, attempt=attempt)
        
        # Filter by relevance score (lower = more similar in pgvector)
        relevant = [(doc, score) for doc, score in results if score < 0.5]
        
        if relevant:
            all_evidence = [
                Evidence(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=round(1 - score, 3)
                )
                for doc, score in relevant
            ]
            print(f"  Found {len(all_evidence)} relevant documents")
            break
        else:
            print(f"  Attempt {attempt}: no relevant evidence found, retrying...")
            attempt += 1
    
    return EvidenceResult(
        claim=claim,
        evidence=all_evidence,
        evidence_found=len(all_evidence) > 0,
        retrieval_attempts=attempt
    )

#    Test    
if __name__ == "__main__":
    test_claims = [
        "The US GDP grew 3.2% in Q2 2024",
        "Unemployment rate fell in October 2024",
        "The Eiffel Tower is located in Berlin",  # should find no evidence
    ]
    
    for claim in test_claims:
        result = hunt_evidence(claim)
        print(f"\nClaim: {claim}")
        print(f"Evidence found: {result.evidence_found}")
        print(f"Attempts needed: {result.retrieval_attempts}")
        for ev in result.evidence:
            print(f"  [{ev.relevance_score}] {ev.source}: {ev.text[:60]}...")
        print("-" * 60)