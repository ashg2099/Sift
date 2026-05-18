import streamlit as st
import requests
import time

API_BASE = "http://localhost:8000"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sift · Claim Verification Engine",
    page_icon="🔍",
    layout="wide",
)

# ── Minimal custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .verdict-true      { background:#d4edda; border-left:4px solid #28a745; padding:12px 16px; border-radius:4px; margin:8px 0; }
    .verdict-false     { background:#f8d7da; border-left:4px solid #dc3545; padding:12px 16px; border-radius:4px; margin:8px 0; }
    .verdict-uncertain { background:#fff3cd; border-left:4px solid #ffc107; padding:12px 16px; border-radius:4px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Sift &nbsp;·&nbsp; Multimodal Claim Verification Engine")
st.markdown(
    "Paste any text — news snippet, social post, report excerpt — and Sift will extract, "
    "retrieve evidence for, and verdict each factual claim using a 4-agent LangGraph pipeline."
)
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
EXAMPLES = {
    "US Economy (2024)": (
        "The US GDP grew 3.2% in Q2 2024. "
        "Unemployment fell to 3.7% in October 2024. "
        "The Federal Reserve kept interest rates at 5.25%."
    ),
    "Climate": (
        "Global temperatures in 2023 were 1.45°C above pre-industrial levels. "
        "Arctic sea ice reached its lowest extent on record in September 2023."
    ),
    "Tech": (
        "OpenAI released GPT-4 in March 2023. "
        "Nvidia became the world's most valuable company briefly in 2024 with a market cap over $3 trillion."
    ),
    "Custom": "",
}

with st.sidebar:
    st.markdown("### ⚡ Quick Examples")
    chosen = st.radio("Load an example:", list(EXAMPLES.keys()), index=3)

    st.divider()
    st.markdown("### ⚙️ Settings")
    poll_interval = st.slider("Poll interval (s)", 2, 10, 3)
    max_wait      = st.slider("Max wait (s)",      30, 300, 200)

    st.divider()
    st.markdown("### 📡 API Status")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            st.success("API online ✅")
        else:
            st.error("API returned error ⚠️")
    except Exception:
        st.error("API offline — start uvicorn + Celery first")

# ── Input ────────────────────────────────────────────────────────────────────
input_text = st.text_area(
    "Text to verify",
    value=EXAMPLES[chosen],
    height=140,
    placeholder="Paste a paragraph containing factual claims…",
)

submit = st.button("🔎 Verify Claims", type="primary", use_container_width=True)

# ── Rendering helpers ─────────────────────────────────────────────────────────
def _verdict_label(decision: str) -> str:
    return {"TRUE": "✅ TRUE", "FALSE": "❌ FALSE", "UNCERTAIN": "⚠️ UNCERTAIN"}.get(decision, decision)


def _css_class(decision: str) -> str:
    return {"TRUE": "verdict-true", "FALSE": "verdict-false"}.get(decision, "verdict-uncertain")


def _confidence_bar(conf: float) -> str:
    pct   = int(conf * 100)
    color = "#28a745" if conf >= 0.7 else "#ffc107" if conf >= 0.5 else "#dc3545"
    return (
        f"<div style='background:#e9ecef;border-radius:4px;height:8px;margin-top:6px;'>"
        f"<div style='width:{pct}%;background:{color};height:8px;border-radius:4px;'></div>"
        f"</div><small>{pct}% confidence</small>"
    )


def render_report(report: dict, idx: int):
    decision  = report.get("decision", "UNCERTAIN")
    claim     = report.get("claim", "—")
    conf      = report.get("final_confidence", report.get("confidence", 0.0))
    reasoning = report.get("revised_reasoning", report.get("reasoning", ""))
    supporting    = report.get("supporting_evidence", [])
    contradicting = report.get("contradicting_evidence", [])
    issues        = report.get("issues_found", [])
    attempts      = report.get("retrieval_attempts", "—")

    st.markdown(
        f"<div class='{_css_class(decision)}'>"
        f"<strong>Claim {idx}:</strong> {claim}<br>"
        f"<span style='font-size:1.1rem;font-weight:700;'>{_verdict_label(decision)}</span>"
        f"{_confidence_bar(conf)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("📄 Full analysis", expanded=False):
        if reasoning:
            st.markdown(f"**Reasoning:** {reasoning}")
        if supporting:
            st.markdown("**Supporting evidence:**")
            for s in supporting:
                st.markdown(f"- {s}")
        if contradicting:
            st.markdown("**Contradicting evidence:**")
            for c in contradicting:
                st.markdown(f"- {c}")
        if issues:
            st.markdown("**Critic notes:**")
            for note in issues:
                st.markdown(f"- ⚠️ {note}")
        st.caption(f"Retrieval attempts: {attempts}")


# ── Main verification flow ────────────────────────────────────────────────────
if submit:
    if not input_text.strip():
        st.warning("Please enter some text to verify.")
        st.stop()

    # Submit
    try:
        resp = requests.post(f"{API_BASE}/verify", json={"text": input_text}, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Make sure uvicorn is running on port 8000.")
        st.stop()
    except Exception as e:
        st.error(f"Submission failed: {e}")
        st.stop()

    task_id = resp.json()["task_id"]
    st.info(f"Task queued · ID: `{task_id}`")

    # Poll loop
    progress_bar = st.progress(0, text="Waiting for agents…")
    elapsed = 0

    with st.spinner("Pipeline running — this takes ~30–90s on first run while the embedding model warms up…"):
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            try:
                sr = requests.get(f"{API_BASE}/status/{task_id}", timeout=10)
                sr.raise_for_status()
                status_data = sr.json()
            except Exception as e:
                st.warning(f"Polling error (will retry): {e}")
                continue

            state   = status_data.get("status", "pending")
            reports = status_data.get("result", {}).get("reports", []) if status_data.get("result") else []

            progress_bar.progress(
                min(elapsed / max_wait, 0.95),
                text=f"Status: **{state}** · {elapsed}s elapsed",
            )

            if state in ("SUCCESS", "complete") and reports:
                progress_bar.progress(1.0, text="✅ Pipeline complete!")

                # Summary metrics
                st.divider()
                st.markdown(f"### Results — {len(reports)} claim(s) analysed")
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ TRUE",      sum(1 for r in reports if r.get("decision") == "TRUE"))
                c2.metric("❌ FALSE",     sum(1 for r in reports if r.get("decision") == "FALSE"))
                c3.metric("⚠️ Uncertain", sum(1 for r in reports if r.get("decision") == "UNCERTAIN"))
                st.divider()

                for i, report in enumerate(reports, 1):
                    render_report(report, i)
                break

            elif state in ("FAILURE", "failed"):
                progress_bar.progress(1.0, text="❌ Task failed")
                st.error(f"Pipeline error: {status_data.get('error', 'unknown')}")
                break

        else:
            progress_bar.warning(f"Timed out after {max_wait}s.")
            st.info(f"Celery may still be running. Poll manually:\n```\nGET {API_BASE}/status/{task_id}\n```")