from dotenv import load_dotenv
load_dotenv()

from celery import Celery
import os
import time
from graph.pipeline import run_sift
import redis

def _get_redis():
    return redis.from_url(REDIS_URL)

# ── Celery App ────────────────────────────────────────
REDIS_URL = os.environ["REDIS_URL"]

# Upstash (rediss://) requires ssl_cert_reqs param for Celery
BROKER_URL = REDIS_URL
if REDIS_URL.startswith("rediss://") and "ssl_cert_reqs" not in REDIS_URL:
    BROKER_URL = REDIS_URL + ("&" if "?" in REDIS_URL else "?") + "ssl_cert_reqs=CERT_NONE"

celery_app = Celery(
    "sift",
    broker=BROKER_URL,
    backend=BROKER_URL
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
        import time
        start = time.time()
        reports = run_sift(text)
        elapsed = round(time.time() - start, 1)

        # Update rolling average in Redis
        try:
            prev = _get_redis().get("sift:stat:avg_latency")
            prev = float(prev) if prev else elapsed
            new_avg = round((prev * 0.8) + (elapsed * 0.2), 1)
            _get_redis().set("sift:stat:avg_latency", new_avg)
        except:
            pass

        return {"status": "complete", "reports": reports}

    except Exception as exc:
        msg = str(exc)
        is_429 = "429" in msg or "rate_limit_exceeded" in msg
        is_tpd = "tokens per day" in msg or "TPD" in msg
        is_tpm = "tokens per minute" in msg or "TPM" in msg

        if is_429 and is_tpd:
            print(f"[TASKS] Daily token limit hit — failing immediately (no retry)")
            raise exc

        if is_429 and (is_tpm or not is_tpd):
            countdown = _rate_limit_wait(exc)
            print(f"[TASKS] Burst rate limit — retrying in {countdown}s")
            raise self.retry(exc=exc, countdown=countdown)

        raise self.retry(exc=exc, countdown=10)