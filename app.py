"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from common.db import checkpoint_saver, db_conn
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


async def load_saved_session(thread_id: str) -> tuple[dict | None, dict | None]:
    """Restore a paused interrupt or completed result from the graph checkpoint."""
    async with checkpoint_saver() as cp:
        await cp.setup()
        app = build_graph(cp)
        snapshot = await app.aget_state({"configurable": {"thread_id": thread_id}})
        payload = snapshot.interrupts[0].value if snapshot.interrupts else None
        values = snapshot.values if isinstance(snapshot.values, dict) else {}
        final = values if payload is None and values.get("final_action") else None
        return payload, final


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
async def load_recent_sessions() -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event,
                   CASE MAX(CASE WHEN risk_level = 'high' THEN 3 WHEN risk_level = 'med' THEN 2 ELSE 1 END)
                        WHEN 3 THEN 'high'
                        WHEN 2 THEN 'med'
                        ELSE 'low'
                   END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 10
            """
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


with st.sidebar:
    st.header("Recent sessions")
    rows = asyncio.run(load_recent_sessions())
    if not rows:
        st.caption("No prior review sessions yet")
    else:
        for row in rows:
            if st.button(
                f"{row['thread_id'][:8]} · {row['pr_url']} · {row['worst_risk']}",
                key=f"thread-{row['thread_id']}",
                use_container_width=True,
            ):
                st.session_state.thread_id = row["thread_id"]
                st.session_state.pr_url = row["pr_url"]
                payload, final = asyncio.run(load_saved_session(row["thread_id"]))
                st.session_state.interrupt_payload = payload
                st.session_state.final = final
                if payload is None and final is None:
                    st.error("This session has no saved graph checkpoint to resume.")
                else:
                    st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    with st.form("approval_form"):
        feedback = st.text_input("Feedback (optional)", key="approval_feedback")
        col1, col2, col3 = st.columns(3)
        approve_clicked = col1.form_submit_button("Approve", type="primary")
        reject_clicked = col2.form_submit_button("Reject")
        edit_clicked = col3.form_submit_button("Edit")

        if approve_clicked:
            return {"choice": "approve", "feedback": feedback}
        if reject_clicked:
            return {"choice": "reject", "feedback": feedback}
        if edit_clicked:
            if not feedback.strip():
                st.error("Describe the changes you want before choosing Edit.")
                return None
            return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers: dict[str, str] = {}
        questions = payload.get("questions", [])
        for index, question in enumerate(questions):
            answers[question] = st.text_input(question, key=f"esc_{index}")
        submitted = st.form_submit_button("Submit answers")
        if submitted:
            unanswered = [question for question in questions if not answers[question].strip()]
            if unanswered:
                st.error("Please answer every escalation question before submitting.")
                return None
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with checkpoint_saver() as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    if kind == "approval_request":
        answer = render_approval_card(payload)
    elif kind == "escalation":
        answer = render_escalation_card(payload)
    else:
        raise ValueError(f"Unknown interrupt kind: {kind!r}")
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    analysis = final.get("analysis")
    if action == "auto_committed" or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    if analysis is not None:
        st.caption(f"Review confidence: {analysis.confidence:.0%}")
        st.markdown(analysis.summary)
    if action == "auto_committed" or action.startswith("committed"):
        st.link_button("View PR on GitHub", st.session_state.pr_url)
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
