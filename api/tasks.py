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
import re

def _rate_limit_wait(exc) -> int:
    """Parse 'Please try again in Xm Ys' from Groq 429 errors → return seconds."""
    msg = str(exc)
    m = re.search(r'try again in\s+(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?', msg, re.IGNORECASE)
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2) or 0)
        return int(minutes * 60 + seconds) + 5   # +5s buffer
    return 30   # default fallback

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
        msg = str(exc)
        is_429 = "429" in msg or "rate_limit_exceeded" in msg
        is_tpd = "tokens per day" in msg or "TPD" in msg   # daily limit — won't recover soon
        is_tpm = "tokens per minute" in msg or "TPM" in msg # burst limit — recovers in seconds

        if is_429 and is_tpd:
            # Daily limit hit — fail immediately, surface error to UI right away
            print(f"[TASKS] Daily token limit hit — failing immediately (no retry)")
            raise exc   # do NOT retry

        if is_429 and (is_tpm or not is_tpd):
            # Burst/minute limit — short wait then retry
            countdown = _rate_limit_wait(exc)
            print(f"[TASKS] Burst rate limit — retrying in {countdown}s")
            raise self.retry(exc=exc, countdown=countdown)

        # All other errors — quick retry
        raise self.retry(exc=exc, countdown=10)