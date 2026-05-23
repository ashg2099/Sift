# 🔍 Sift — Multimodal Claim Verification Engine

> A production-grade, multi-agent fact-checking pipeline that extracts claims from any text, retrieves grounded evidence, and delivers auditable verdicts with cited sources.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green)
![pgvector](https://img.shields.io/badge/pgvector-hybrid--search-purple)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## What it does

Paste any text — a news article, a social media post, a politician's speech — and Sift will:

1. **Extract** every distinct factual claim using structured LLM output
2. **Retrieve** real evidence via Tavily web search, Guardian API, and pgvector RAG
3. **Synthesize** a verdict (TRUE / FALSE / UNCERTAIN) grounded strictly in retrieved evidence
4. **Critique** that verdict adversarially — checking for overconfidence, unsupported reasoning, and alternative interpretations
5. **Correct** FALSE/UNCERTAIN claims by finding what is actually true
6. **Return** a structured report with cited sources, confidence scores (shown as a visual ring), and full reasoning chains

Results are cached in Redis for 7 days — repeated claims return instantly.

---

## Architecture

```
Input Text
    │
    ▼
┌─────────────────┐
│ Claim Extractor │  LLaMA 3.3 70B + Pydantic structured output
└────────┬────────┘
         │ List[Claim]
         ▼
┌─────────────────┐     ┌──────────────────────────────────┐
│ Evidence Hunter │────▶│  Tavily (web) + Guardian + NewsAPI│
│   (HyDE RAG)    │     │  + pgvector (bge-m3 embeddings)   │
└────────┬────────┘     └──────────────────────────────────┘
         │ Evidence[]
         ▼
┌─────────────────┐
│ Synthesis Agent │  Verdict: TRUE / FALSE / UNCERTAIN + confidence
└────────┬────────┘
         │ Verdict
         ▼
┌─────────────────┐
│   Critic Agent  │  Adversarial review — adjusts confidence + flags issues
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Correction Agent│  On FALSE/UNCERTAIN — finds what is actually true
└────────┬────────┘
         │ Final Report
         ▼
    LangGraph State ──▶ loops over all claims ──▶ results[]
```

**Infrastructure:**
- **FastAPI** — async REST API, returns task ID immediately
- **Celery + Redis** — async task queue, pipeline runs in background
- **Redis Cache** — 7-day cache, repeated claims return in <1s
- **pgvector (Postgres)** — vector similarity search on evidence chunks
- **LangFuse** — LLM observability and tracing
- **Prometheus + Grafana** — metrics and monitoring

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | `BAAI/bge-m3` via HuggingFace Inference API |
| Orchestration | LangGraph state machine |
| RAG | HyDE + pgvector hybrid search |
| Vector DB | PostgreSQL + pgvector extension |
| API | FastAPI + Pydantic |
| Task Queue | Celery + Redis |
| Evidence Sources | Tavily (primary) + Guardian API + NewsAPI |
| Observability | LangFuse + Prometheus + Grafana |
| UI | Custom HTML/JS (dark theme) |

---

## Quick Start (Docker)

The easiest way to run Sift — one command starts everything.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Groq API key](https://console.groq.com) (free)
- [Tavily API key](https://app.tavily.com) (free — 1,000 searches/month)
- [HuggingFace token](https://huggingface.co/settings/tokens) (free)

### Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/sift.git
cd sift

# 2. Add your API keys
cp .env.example .env
# Open .env and fill in your keys

# 3. Start everything
docker compose up
```

Open **http://localhost:8000** and start verifying claims.

That's it. Docker starts Postgres, Redis, the FastAPI server, the Celery worker, Prometheus, and Grafana all at once.

### Optional: Ingest evidence into pgvector

The pipeline works without this step (it falls back to live web search). But ingesting articles improves retrieval quality for news-related claims:

```bash
docker compose exec worker python -m ingestion.loaders
```

---

## API Usage

```bash
# Submit text for verification
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "The Eiffel Tower is located in Paris."}'

# {"task_id": "abc123", "status": "queued"}

# Poll for results
curl http://localhost:8000/status/abc123
```

---

## Project Structure

```
Sift/
├── agents/
│   ├── claim_extractor.py   # Agent 1: extract structured claims
│   ├── evidence_hunter.py   # Agent 2: HyDE retrieval + live search
│   ├── synthesis.py         # Agent 3: verdict synthesis
│   ├── critic.py            # Agent 4: adversarial review
│   └── corrector.py         # Agent 5: find what is actually true
├── graph/
│   └── pipeline.py          # LangGraph state machine + Redis cache
├── api/
│   ├── main.py              # FastAPI endpoints + API-layer cache
│   └── tasks.py             # Celery task definitions
├── ingestion/
│   └── loaders.py           # Guardian + Wikipedia ingestion
├── ui/
│   └── index.html           # Dark theme UI with claim highlighting
├── monitoring/
│   └── prometheus.yml
├── docker-compose.yml       # Full stack — one command setup
├── Dockerfile
├── requirements.txt         # Dev dependencies (includes local model)
└── requirements-prod.txt    # Prod dependencies (HF Inference API)
```

---

## Key Design Decisions

**Why HyDE?** Standard RAG embeds the raw claim and searches for similar text. HyDE generates a *hypothetical document* that would contain the evidence, then embeds that — producing richer semantic signal and significantly better retrieval recall on short factual claims.

**Why a Critic agent?** The synthesis agent tends toward overconfidence when evidence partially supports a claim. The critic is prompted adversarially to find unsupported statements, flag alternative interpretations, and adjust confidence downward when warranted. This prevents false certainty.

**Why a Correction agent?** Knowing something is false isn't enough — users need to know what is actually true. The correction agent runs on FALSE/UNCERTAIN verdicts and finds the correct information using live search + model knowledge.

**Why Celery?** Verification takes 10–60s per run depending on claim count. A synchronous API would block. Celery decouples submission from result retrieval — the API returns a task ID immediately, and the client polls.

**Why Redis cache?** The same claims get submitted repeatedly (especially viral misinformation). Caching at both the API layer and pipeline layer means repeated claims return in <1s with zero LLM cost.

---

## Roadmap

- [ ] Visual claim verification (image-based claims)
- [ ] URL input — paste an article URL and verify all claims in it
- [ ] Export results as PDF report
- [ ] GitHub Actions CI/CD
- [ ] 20-claim accuracy benchmark

---

## License

MIT
