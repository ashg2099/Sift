from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, validator
from typing import List, Literal

class Claim(BaseModel):
    text: str
    claim_type: Literal["statistical", "causal", "entity", "temporal"]
    checkable: bool   # can this actually be verified?

    @validator("checkable", pre=True)
    def coerce_bool(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() not in ("false", "0", "no", "")
        return bool(v)

class ClaimList(BaseModel):
    claims: List[Claim]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(ClaimList)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract every verifiable factual claim from the text.

Rules:
- Extract EACH distinct factual assertion as its own separate claim
- Do NOT merge or deduplicate claims — even if two claims relate to the same event, keep them separate
- A date/time claim and an outcome claim about the same event are TWO different claims
  Example: "The election was held on Nov 5th" and "Trump won the election" → 2 separate claims
- Include claims about elections, people, statistics, locations, events, dates, and outcomes
- Only skip genuine opinions, predictions, or purely subjective statements
- If a sentence contains multiple verifiable facts, extract each one individually
- Political facts (who won an election, who held office) are always checkable"""),
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