from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Literal

class Claim(BaseModel):
    text: str
    claim_type: Literal["statistical", "causal", "entity", "temporal"]
    checkable: bool   # can this actually be verified?

class ClaimList(BaseModel):
    claims: List[Claim]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(ClaimList)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract every verifiable factual claim from the text.
Only extract objective claims that can be confirmed or denied with evidence.
Do not extract opinions, predictions, or subjective statements."""),
    ("human", "{text}")
])

extractor = prompt | structured_llm

def extract_claims(text: str) -> ClaimList:
    return extractor.invoke({"text": text})

# Test it directly
if __name__ == "__main__":
    sample = """
    The US Federal Reserve raised interest rates by 0.25% in March 2024.
    Analysts believe this will slow inflation, which hit 3.2% in February.
    The stock market fell 1.4% following the announcement.
    """
    result = extract_claims(sample)
    for claim in result.claims:
        print(f"[{claim.claim_type}] {claim.text}")