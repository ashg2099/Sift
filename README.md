# 🔍 Sift — Multimodal Claim Verification Engine

> A production-grade, multi-agent fact-checking pipeline that extracts claims from any text, retrieves grounded evidence, and delivers auditable verdicts with cited sources.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green)
![pgvector](https://img.shields.io/badge/pgvector-hybrid--search-purple)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-red)

---

## What it does

Paste any text — a news article, a social media post, a politician's speech — and Sift will:

1. **Extract** every distinct factual claim using structured LLM output
2. **Retrieve** real evidence using HyDE (Hypothetical Document Embeddings) against a pgvector store of Guardian and Wikipedia articles
3. **Synthesize** a verdict (TRUE / FALSE / UNCERTAIN) grounded strictly in retrieved evidence
4. **Critique** that verdict adversarially — checking for overconfidence, unsupported reasoning, and alternative interpretations
5. **Return** a structured report with cited sources, confidence scores, and full reasoning chains

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
┌─────────────────┐     ┌──────────────────────────┐
│ Evidence Hunter │────▶│  pgvector (bge-m3 embeds) │
│   (HyDE RAG)    │     │  Guardian API + Wikipedia  │
└────────┬────────┘     └──────────────────────────┘
         │ Evidence[]
         ▼
┌─────────────────┐
│ Synthesis Agent │  Verdict: TRUE / FALSE / UNCERTAIN
└────────┬────────┘
         │ Verdict
         ▼
┌─────────────────┐
│   Critic Agent  │  Adversarial review — adjusts confidence
└────────┬────────┘
         │ Final Report
         ▼
    LangGraph State ──▶ loops over all claims ──▶ results[]
```

**Infrastructure:**
- **FastAPI** — async REST API, returns task ID immediately
- **Celery + Redis** — async task queue, pipeline runs in background
- **pgvector (Postgres)** — vector similarity search on 4,200+ evidence chunks
- **Prometheus + Grafana** — metrics and observability
- **Streamlit** — interactive demo UI

---

## Eval Results

Benchmarked on 20 labelled claims across TRUE, FALSE, and UNCERTAIN categories:

| Metric | Score |
|--------|-------|
| Overall Accuracy | **64.7%** |
| TRUE detection | **83.3%** |
| FALSE detection | **60.0%** |
| UNCERTAIN detection | **50.0%** |
| Avg Confidence | **58.9%** |
| Avg Latency | **11.2s / claim** |

> The system is deliberately conservative — it prefers UNCERTAIN over FALSE when evidence is ambiguous, which is the correct behaviour for a fact-checking tool.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | `BAAI/bge-m3` (multilingual, 2.27GB) |
| Orchestration | LangGraph state machine |
| RAG | HyDE + pgvector hybrid search |
| Vector DB | PostgreSQL + pgvector extension |
| API | FastAPI + Pydantic |
| Task Queue | Celery + Redis |
| Evidence Sources | Guardian API + Wikipedia |
| Observability | Prometheus + Grafana |
| UI | Streamlit |

---

## Project Structure

```
Sift/
├── agents/
│   ├── claim_extractor.py   # Agent 1: extract structured claims
│   ├── evidence_hunter.py   # Agent 2: HyDE retrieval
│   ├── synthesis.py         # Agent 3: verdict synthesis
│   └── critic.py            # Agent 4: adversarial review
├── graph/
│   └── pipeline.py          # LangGraph state machine
├── retrieval/
│   └── vector_store.py      # pgvector setup + search
├── ingestion/
│   └── loaders.py           # Guardian + Wikipedia ingestion
├── api/
│   ├── main.py              # FastAPI endpoints
│   └── tasks.py             # Celery task definitions
├── eval/
│   └── benchmark.py         # RAGAS + accuracy evaluation
├── ui/
│   └── app.py               # Streamlit demo
├── monitoring/
│   └── prometheus.yml
├── docker-compose.yml        # Postgres, Redis, Prometheus, Grafana
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- [Groq API key](https://console.groq.com) (free)
- [Guardian API key](https://open-platform.theguardian.com/access/) (free)

### Setup

```bash
# 1. Clone and create virtualenv
git clone https://github.com/yourusername/sift.git
cd sift
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start infrastructure
docker compose up -d

# 5. Ingest evidence (one-time, ~7 mins)
PYTHONPATH=. python -m ingestion.loaders

# 6. Start services
# Terminal 1 — Celery worker
PYTHONPATH=. celery -A api.tasks worker --loglevel=info

# Terminal 2 — FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 3 — Streamlit UI
streamlit run ui/app.py
```

Open **http://localhost:8501** and start verifying claims.

### API Usage

```bash
# Submit a claim for verification
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "The US GDP grew 3.2% in Q2 2024."}'

# {"task_id": "abc123", "status": "queued"}

# Poll for results
curl http://localhost:8000/status/abc123
```

---

## Key Design Decisions

**Why HyDE?** Standard RAG embeds the raw claim and searches for similar text. HyDE generates a *hypothetical document* that would contain the evidence, then embeds that — producing richer semantic signal and significantly better retrieval recall on short factual claims.

**Why a Critic agent?** The synthesis agent tends toward overconfidence when evidence partially supports a claim. The critic is prompted adversarially to find unsupported statements, flag alternative interpretations, and adjust confidence downward when warranted. This prevents false certainty.

**Why Celery?** Verification takes 10–90s per run depending on claim count. A synchronous API would block. Celery decouples submission from result retrieval — the API returns a task ID immediately, and the client polls.

---

## Roadmap

- [ ] Visual claim verification (PaliGemma VQA for image-based claims)
- [ ] Live evidence retrieval at query time (Guardian API on-demand)
- [ ] Deploy to Fly.io + Neon + Upstash
- [ ] Arize Phoenix observability traces
- [ ] GitHub Actions CI/CD

---

## License

MIT
