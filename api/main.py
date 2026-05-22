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
    task = verify_text_task.delay(request.text)
    return TaskResponse(
        task_id=task.id,
        status="queued",
        message="Verification started. Poll /status/{task_id} for results."
    )

@app.get("/status/{task_id}", response_model=StatusResponse)
def get_status(task_id: str):
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