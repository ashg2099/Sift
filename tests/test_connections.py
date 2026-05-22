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

def test_langfuse():
    from langfuse import Langfuse
    client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    result = client.auth_check()
    assert result, "LangFuse auth failed — check your keys"
    print("✓ LangFuse: connected. Check cloud.langfuse.com")

if __name__ == "__main__":
    test_postgres()
    test_redis()
    test_groq()
    test_langfuse()
    print("\nAll connections working!")