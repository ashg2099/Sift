import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def get_vector_store():
    return PGVector(
        embeddings=embedder,
        collection_name="veridai_evidence",
        connection=os.environ["DATABASE_URL"],
    )

def add_documents(docs: list[Document]):
    store = get_vector_store()
    store.add_documents(docs)
    print(f"Added {len(docs)} documents to vector store")

def search(query: str, k: int = 5) -> list[Document]:
    store = get_vector_store()
    return store.similarity_search(query, k=k)

# Test: add 3 fake documents and search
if __name__ == "__main__":
    test_docs = [
        Document(page_content="US GDP grew 3.2% in Q2 2024 according to the Bureau of Economic Analysis.",
                 metadata={"source": "bea.gov", "date": "2024-07-25"}),
        Document(page_content="Federal Reserve minutes show inflation target remains at 2% annually.",
                 metadata={"source": "federalreserve.gov", "date": "2024-08-01"}),
        Document(page_content="Unemployment rate fell to 3.7% in October 2024, matching analyst forecasts.",
                 metadata={"source": "bls.gov", "date": "2024-11-01"}),
    ]
    add_documents(test_docs)
    results = search("economic growth GDP")
    for doc in results:
        print(f"\n[{doc.metadata['source']}] {doc.page_content[:80]}...")