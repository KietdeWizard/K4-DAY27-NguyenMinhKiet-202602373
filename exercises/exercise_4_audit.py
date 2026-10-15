"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

Approach:
    - Every meaningful event is written as an AuditEntry.
    - Approval and escalation pending events are separate nodes before their
      interrupts, so resuming an interrupt cannot duplicate those audit rows.
    - The durable checkpointer uses a scoped serializer for the review models.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import checkpoint_saver, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself.
    """
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]> fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]> analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": "Senior reviewer. Structured output."},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]OK[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]> route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision=decision,
        reason=f"Route selected {decision} at confidence {c:.0%}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"decision": decision}


async def node_approval_pending(state):
    a = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="Waiting for human approval decision",
        execution_time_ms=0,
    ))
    return {}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]
    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    choice = resp.get("choice")
    if choice not in {"approve", "reject", "edit"}:
        raise ValueError(f"Invalid approval choice: {choice!r}")
    feedback = resp.get("feedback", "").strip()
    if choice == "edit" and not feedback:
        raise ValueError("Feedback is required when editing the review")
    reviewer_id = os.environ.get("GITHUB_USER")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=reviewer_id,
        decision=choice,
        reason=(feedback or f"Human chose {choice}"),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"human_choice": choice, "human_feedback": feedback}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> str:
    post_review_comment(state["pr_url"], _render_comment_body(state))
    console.print(f"  [green]OK[/green] posted comment to {state['pr_url']}")
    return "committed"


async def node_edit_review(state):
    """Revise the proposed review using the reviewer's requested changes."""
    t0 = time.monotonic()
    a = state["analysis"]
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]Revising review from reviewer feedback...[/dim]"):
        revised: PRAnalysis = await llm.ainvoke([
            {
                "role": "system",
                "content": "Revise the proposed code review according to the reviewer feedback. Preserve valid findings and return a structured review.",
            },
            {
                "role": "user",
                "content": (
                    f"PR title: {state['pr_title']}\n\n"
                    f"Diff:\n{state['pr_diff']}\n\n"
                    f"Original review:\n{a.model_dump_json(indent=2)}\n\n"
                    f"Reviewer feedback:\n{state.get('human_feedback', '')}"
                ),
            },
        ])
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=revised.confidence,
        risk_level=risk_level_for(revised.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="edit",
        reason="Revised review using reviewer feedback",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": revised}


async def node_commit(state):
    console.print("[cyan]> commit[/cyan]")
    t0 = time.monotonic()
    escalation_answers = state.get("escalation_answers")
    human_choice = state.get("human_choice")
    if human_choice in {"approve", "edit"}:
        _post(state)
        action = "committed_edited" if human_choice == "edit" else "committed"
        decision = human_choice
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={human_choice})")
        action = "rejected"
        decision = "reject"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=state["analysis"].confidence,
        risk_level=risk_level_for(state["analysis"].confidence),
        reviewer_id=os.environ.get("GITHUB_USER") if human_choice or escalation_answers is not None else None,
        decision=decision,
        reason=(
            f"Final action after escalation review was {action}"
            if escalation_answers is not None
            else f"Final action was {action}"
        ),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": action}


async def node_auto_approve(state):
    console.print("[cyan]> auto_approve[/cyan]  [dim]high confidence - posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action = _post(state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="auto",
        reason=f"Auto-approved review at confidence {a.confidence:.0%}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": f"auto_{action}"}


async def node_escalation_pending(state):
    a = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="Waiting for reviewer answers to escalation questions",
        execution_time_ms=0,
    ))
    return {}


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]
    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    if not isinstance(answers, dict) or not all(
        isinstance(answers.get(question), str) and answers[question].strip()
        for question in questions
    ):
        raise ValueError("A non-empty answer is required for every escalation question")

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="escalate",
        reason="Reviewer answered escalation questions",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]> synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {
                "role": "system",
                "content": "Re-evaluate and refine the structured code review using the original analysis and reviewer answers. Do not invent facts; preserve unresolved risks.",
            },
            {
                "role": "user",
                "content": (
                    f"PR title: {state['pr_title']}\n\n"
                    f"Diff:\n{state['pr_diff']}\n\n"
                    f"Original analysis:\n{state['analysis'].model_dump_json(indent=2)}\n\n"
                    f"Reviewer Q&A:\n{qa}"
                ),
            },
        ])
    console.print(f"  [green]OK[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="escalate",
        reason=f"Re-synthesized review after escalation questions (confidence {refined.confidence:.0%})",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": refined}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("approval_pending", node_approval_pending),
        ("human_approval", node_human_approval), ("edit_review", node_edit_review),
        ("commit", node_commit), ("escalation_pending", node_escalation_pending),
        ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "approval_pending", "escalate": "escalation_pending"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("approval_pending", "human_approval")
    g.add_conditional_edges(
        "human_approval",
        lambda state: "edit_review" if state["human_choice"] == "edit" else "commit",
        {"edit_review": "edit_review", "commit": "commit"},
    )
    g.add_edge("edit_review", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalation_pending", "escalate")
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "approval_pending")
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"], title=f"conf={payload['confidence']:.0%}", border_style="green",
        ))
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        feedback = console.input("Feedback: ").strip() if choice != "approve" else ""
        while choice == "edit" and not feedback:
            feedback = console.input("Describe the requested edits: ").strip()
        return {"choice": choice, "feedback": feedback}
    if payload["kind"] == "escalation":
        answers = {}
        for question in payload["questions"]:
            answer = ""
            while not answer:
                answer = console.input(f"Q: {question}\nA: ").strip()
            answers[question] = answer
        return answers
    raise ValueError(f"Unknown interrupt kind: {payload['kind']!r}")


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with checkpoint_saver() as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
