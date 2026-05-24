from dotenv import load_dotenv
load_dotenv()

import os
import time
import requests

import wikipedia
wikipedia.set_user_agent("Sift/1.0 (fact-checking research tool)")

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =================== Vector store setup ===================
embedder = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HF_TOKEN", ""),
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
            time.sleep(0.5)
        except Exception as e:
            print(f"  Guardian error for '{topic}' page {page}: {e}")
    return docs


# =================== NewsAPI ===================
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Topics intentionally different from Guardian to avoid overlap
NEWSAPI_TOPICS = [
    "OpenAI ChatGPT artificial intelligence",
    "Elon Musk Tesla SpaceX",
    "Ukraine Russia war",
    "China economy trade tariffs",
    "cryptocurrency Bitcoin Ethereum",
    "US housing market mortgage rates",
    "renewable energy solar wind power",
    "US healthcare Medicare Medicaid",
    "immigration border policy",
    "COVID pandemic health WHO",
    "Nvidia GPU semiconductor",
    "Amazon Apple Microsoft earnings",
]

def fetch_newsapi(topic: str, page_size: int = 20) -> list[Document]:
    """
    Fetch articles from NewsAPI for a topic.
    Note: free tier truncates article content to ~200 chars,
    so we combine title + description + content for richer chunks.
    """
    docs = []
    if not NEWSAPI_KEY:
        print("  ⚠ NEWSAPI_KEY not set — skipping NewsAPI ingestion")
        return docs

    params = {
        "q": topic,
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
    }
    try:
        r = requests.get(NEWSAPI_URL, params=params, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        for article in articles:
            title       = article.get("title", "") or ""
            description = article.get("description", "") or ""
            content     = article.get("content", "") or ""
            url         = article.get("url", "")
            source_name = article.get("source", {}).get("name", "newsapi")

            # Combine all available text (content is truncated ~200 chars on free tier)
            full_text = " ".join(filter(None, [title, description, content])).strip()
            if len(full_text) < 60:
                continue

            chunks = splitter.split_text(full_text)
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "headline": title,
                        "provider": "newsapi",
                        "source_name": source_name,
                    }
                ))
        time.sleep(0.3)
    except Exception as e:
        print(f"  NewsAPI error for '{topic}': {e}")
    return docs


# =================== Wikipedia ===================
WIKI_TOPICS = [
    # Economics
    "United States GDP",
    "Federal Reserve",
    "Unemployment in the United States",
    "2024 United States presidential election",
    "Inflation",
    "United States housing bubble",
    # Climate
    "Climate change",
    "Global warming",
    "Arctic sea ice decline",
    # AI & Big Tech
    "GPT-4",
    "ChatGPT",
    "OpenAI",
    "Nvidia",
    "Nvidia market capitalization",
    "Artificial intelligence",
    "Apple Inc.",
    "Microsoft",
    "Amazon (company)",
    # Geopolitics
    "Ukraine war",
    "Russo-Ukrainian War",
    # Finance & Crypto
    "Bitcoin",
    "Cryptocurrency",
    # Other
    "SpaceX",
    "COVID-19 pandemic",
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

    if NEWSAPI_KEY:
        print("\n🗞  Fetching from NewsAPI...")
        for topic in NEWSAPI_TOPICS:
            print(f"  → {topic}")
            docs = fetch_newsapi(topic, page_size=20)
            print(f"     {len(docs)} chunks")
            all_docs.extend(docs)
    else:
        print("\n⚠  NEWSAPI_KEY not set — skipping NewsAPI (add it to .env to include)")

    print("\n📖 Fetching from Wikipedia...")
    for topic in WIKI_TOPICS:
        print(f"  → {topic}")
        docs = fetch_wikipedia(topic)
        print(f"     {len(docs)} chunks")
        all_docs.extend(docs)
        time.sleep(1.5)

    print(f"\n✅ Total chunks to embed: {len(all_docs)}")
    print("⏳ Embedding and storing in pgvector (this takes a few minutes)...")
    
    print("🗑  Clearing existing vectors...")
    vector_store.delete_collection()
    vector_store.create_collection()

    batch_size = 10
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"  Stored batch {i // batch_size + 1}/{(len(all_docs) // batch_size) + 1}")
        time.sleep(0.5)

    print(f"\n🎉 Done! {len(all_docs)} chunks ingested into pgvector.")


if __name__ == "__main__":
    ingest_all()