from dotenv import load_dotenv
load_dotenv()

from celery import Celery
import os

# ── Celery App ────────────────────────────────────────
celery_app = Celery(
    "sift",
    broker=os.environ["REDIS_URL"],
    backend=os.environ["REDIS_URL"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,     # results kept for 1 hour
)

# ── Task ──────────────────────────────────────────────
@celery_app.task(bind=True, max_retries=3)
def verify_text_task(self, text: str):
    try:
        from graph.pipeline import run_sift
        reports = run_sift(text)
        return {
            "status": "complete",
            "reports": reports
        }
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)