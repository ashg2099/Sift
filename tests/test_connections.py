import os
from dotenv import load_dotenv
load_dotenv()

def test_groq():
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":"Say: GROQ OK"}],
        max_tokens=10
    )
    print("✓ Groq:", r.choices[0].message.content)

def test_postgres():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SELECT version();")
    print("✓ Postgres:", cur.fetchone()[0][:30])
    conn.close()

def test_redis():
    import redis
    r = redis.from_url(os.environ["REDIS_URL"])
    r.set("test", "REDIS OK")
    print("✓ Redis:", r.get("test").decode())

def test_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", max_tokens=10)
    r = llm.invoke("Say: LANGSMITH OK")
    print("✓ LangSmith trace sent. Check smith.langchain.com")

if __name__ == "__main__":
    test_postgres()
    test_redis()
    test_groq()
    test_langsmith()
    print("\nAll connections working!")