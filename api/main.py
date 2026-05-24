from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from api.tasks import verify_text_task, celery_app
from prometheus_client import Counter, Histogram, make_asgi_app
import uuid
import os
import json
import hashlib
import redis as redis_lib

# ── Shared Redis client ───────────────────────────────
_redis = None

def _get_redis():
    global _redis
    if _redis is None:
        _redis = redis_lib.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    return _redis

def _cache_key(text: str) -> str:
    """Must match the key used in graph/pipeline.py exactly."""
    normalised = text.lower().strip().rstrip('.')
    return f"sift:v1:{hashlib.md5(normalised.encode()).hexdigest()}"

INSTANT_TTL = 60 * 5   # 5 minutes — just long enough for the UI to poll

# ── Prometheus Metrics ────────────────────────────────
claims_total = Counter(
    "sift_claims_total",
    "Total verification tasks submitted"
)
task_duration = Histogram(
    "sift_task_duration_seconds",
    "Time spent on verification tasks",
    buckets=[1, 2, 5, 10, 30, 60]
)

# ── FastAPI App ───────────────────────────────────────
app = FastAPI(title="Sift", version="1.0.0",
              description="Multimodal Claim Verification Engine")

# Mount prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Mount static UI files
ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")

# ── Request/Response Models ───────────────────────────
class VerifyRequest(BaseModel):
    text: str

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict | None = None

# ── Endpoints ─────────────────────────────────────────
@app.get("/")
def serve_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "..", "ui", "index.html")
    return FileResponse(ui_path)

@app.get("/health")
def health():
    return {"status": "ok", "service": "Sift API"}

@app.post("/verify", response_model=TaskResponse)
def verify(request: VerifyRequest):
    claims_total.inc()

    # ── Cache check — skip Celery entirely on hit ─────
    key = _cache_key(request.text)
    try:
        cached = _get_redis().get(key)
        if cached:
            # Store result under a short-lived instant key so /status can find it
            instant_id = f"instant::{uuid.uuid4().hex}"
            payload = json.dumps({"status": "complete", "reports": json.loads(cached)})
            _get_redis().setex(f"sift:instant:{instant_id}", INSTANT_TTL, payload)
            print(f"[API] Cache HIT — instant task_id: {instant_id[:24]}...")
            return TaskResponse(
                task_id=instant_id,
                status="queued",
                message="Verification started. Poll /status/{task_id} for results."
            )
    except Exception as e:
        print(f"[API] Cache check error (falling back to Celery): {e}")

    # ── Cache miss — dispatch to Celery as normal ─────
    task = verify_text_task.delay(request.text)
    return TaskResponse(
        task_id=task.id,
        status="queued",
        message="Verification started. Poll /status/{task_id} for results."
    )

@app.get("/status/{task_id}", response_model=StatusResponse)
def get_status(task_id: str):
    # ── Instant (cached) task — no Celery involved ────
    if task_id.startswith("instant::"):
        try:
            raw = _get_redis().get(f"sift:instant:{task_id}")
            if raw:
                data = json.loads(raw)
                return StatusResponse(
                    task_id=task_id,
                    status="complete",
                    result={"status": "complete", "reports": data["reports"]}
                )
        except Exception as e:
            print(f"[API] Instant task lookup error: {e}")
        # If key expired or missing, fall through to a not-found response
        return StatusResponse(task_id=task_id, status="failed",
                              result={"error": "Cached result expired — please resubmit."})

    # ── Normal Celery task ────────────────────────────
    task = verify_text_task.AsyncResult(task_id)

    if task.state == "PENDING":
        return StatusResponse(task_id=task_id, status="pending")

    elif task.state == "STARTED":
        return StatusResponse(task_id=task_id, status="processing")

    elif task.state == "SUCCESS":
        return StatusResponse(
            task_id=task_id,
            status="complete",
            result=task.result
        )

    elif task.state == "FAILURE":
        return StatusResponse(
            task_id=task_id,
            status="failed",
            result={"error": str(task.result)}
        )

    return StatusResponse(task_id=task_id, status=task.state.lower())

@app.get("/stats")
def get_stats():
    # Evidence count from pgvector
    chunk_count = 0
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(os.environ["DATABASE_URL"])
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding"))
            chunk_count = result.scalar() or 0
    except Exception as e:
        print(f"[STATS] DB error: {e}")

    # Avg latency from Redis
    try:
        avg_latency = _get_redis().get("sift:stat:avg_latency")
        avg_latency = float(avg_latency) if avg_latency else 11.2
    except:
        avg_latency = 11.2

    # Accuracy — manually set
    try:
        accuracy = _get_redis().get("sift:stat:accuracy")
        accuracy = float(accuracy) if accuracy else 83.3
    except:
        accuracy = 83.3

    return {"chunks": chunk_count, "latency": round(avg_latency, 1), "accuracy": accuracy}