from dotenv import load_dotenv
load_dotenv()

import os
import time
import requests
import wikipedia
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =================== Vector store setup ===================
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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# =================== Guardian API ===================
GUARDIAN_KEY = os.environ["GUARDIAN_API_KEY"]
GUARDIAN_URL = "https://content.guardianapis.com/search"

GUARDIAN_TOPICS = [
    "US economy GDP",
    "Federal Reserve interest rates",
    "unemployment jobs",
    "climate change temperature",
    "artificial intelligence",
    "stock market",
    "inflation consumer prices",
    "US elections",
]

def fetch_guardian(topic: str, pages: int = 2) -> list[Document]:
    docs = []
    for page in range(1, pages + 1):
        params = {
            "q": topic,
            "api-key": GUARDIAN_KEY,
            "show-fields": "bodyText,headline",
            "page-size": 10,
            "page": page,
        }
        try:
            r = requests.get(GUARDIAN_URL, params=params, timeout=10)
            r.raise_for_status()
            results = r.json().get("response", {}).get("results", [])
            for item in results:
                fields = item.get("fields", {})
                body = fields.get("bodyText", "")
                headline = fields.get("headline", "")
                url = item.get("webUrl", "")
                if body:
                    chunks = splitter.split_text(body)
                    for chunk in chunks:
                        docs.append(Document(
                            page_content=chunk,
                            metadata={"source": url, "headline": headline, "provider": "guardian"}
                        ))
            time.sleep(0.5)  # be polite to the API
        except Exception as e:
            print(f"  Guardian error for '{topic}' page {page}: {e}")
    return docs


# =================== Wikipedia ===================
WIKI_TOPICS = [
    "United States GDP",
    "Federal Reserve",
    "Unemployment in the United States",
    "2024 United States presidential election",
    "Climate change",
    "Inflation",
    "Artificial intelligence",
    "OpenAI",
    "Nvidia",
    "Ukraine war",
]

def fetch_wikipedia(topic: str) -> list[Document]:
    docs = []
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        chunks = splitter.split_text(page.content)
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"source": page.url, "headline": topic, "provider": "wikipedia"}
            ))
    except wikipedia.exceptions.DisambiguationError as e:
        # Take the first option if ambiguous
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            chunks = splitter.split_text(page.content)
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": page.url, "headline": topic, "provider": "wikipedia"}
                ))
        except Exception as inner:
            print(f"  Wikipedia disambiguation fallback failed for '{topic}': {inner}")
    except Exception as e:
        print(f"  Wikipedia error for '{topic}': {e}")
    return docs


# =================== Ingest into pgvector ===================
def ingest_all():
    all_docs = []

    print("\n📰 Fetching from Guardian API...")
    for topic in GUARDIAN_TOPICS:
        print(f"  → {topic}")
        docs = fetch_guardian(topic, pages=2)
        print(f"     {len(docs)} chunks")
        all_docs.extend(docs)

    print("\n📖 Fetching from Wikipedia...")
    for topic in WIKI_TOPICS:
        print(f"  → {topic}")
        docs = fetch_wikipedia(topic)
        print(f"     {len(docs)} chunks")
        all_docs.extend(docs)

    print(f"\n✅ Total chunks to embed: {len(all_docs)}")
    print("⏳ Embedding and storing in pgvector (this takes a few minutes)...")

    # Store in batches of 50 to avoid memory spikes
    batch_size = 50
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"  Stored batch {i // batch_size + 1}/{(len(all_docs) // batch_size) + 1}")

    print(f"\n🎉 Done! {len(all_docs)} chunks ingested into pgvector.")


if __name__ == "__main__":
    ingest_all()