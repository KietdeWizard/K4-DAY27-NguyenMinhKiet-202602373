"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 60%, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import checkpoint_serializer
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state):
    console.print("[cyan]> fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state):
    console.print("[cyan]> analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": (
                "Senior reviewer. Structured output. "
                "If confidence is below 0.60, populate escalation_questions with 2–4 specific, "
                "context-rich questions that reference the exact file, function, or diff section you are uncertain about. "
                "Ask about intent, security impact, migration risk, or missing validation/tests."
            )},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]OK[/green] confidence={analysis.confidence:.0%}, {len(analysis.escalation_questions)} question(s)")
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]> route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD: decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:    decision = "escalate"
    else:                           decision = "human_approval"
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
    a = state["analysis"]
    questions = a.escalation_questions
    if not questions:
        questions = ["What is the intent of this PR?", "Any migration concerns?"]

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
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    qa = "\n".join(f"Q: {q}\nA: {ans}" for q, ans in (state.get("escalation_answers") or {}).items())
    initial = state["analysis"]
    llm = get_llm().with_structured_output(PRAnalysis)
    refined = llm.invoke([
        {"role": "system", "content": "Refine the review using the reviewer answers. Keep the output structured and base the new confidence on the remaining uncertainty."},
        {"role": "user", "content": (
            f"Original PR title: {state.get('pr_title', '')}\n\n"
            f"Original diff:\n{state['pr_diff']}\n\n"
            f"Initial summary:\n{initial.summary}\n\n"
            f"Initial risk factors:\n{', '.join(initial.risk_factors) if initial.risk_factors else 'None'}\n\n"
            f"Initial confidence: {initial.confidence:.0%}\n\n"
            f"Reviewer Q&A:\n{qa}\n"
            "Using the answers, re-evaluate the review. If uncertainty remains, explain it in confidence_reasoning and ask targeted follow-up questions only if truly necessary."
        )},
    ])
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    choice = response.get("choice")
    if choice not in {"approve", "reject", "edit"}:
        raise ValueError(f"Invalid approval choice: {choice!r}")
    feedback = response.get("feedback", "").strip()
    if choice == "edit" and not feedback:
        raise ValueError("Feedback is required when editing the review")
    return {"human_choice": choice, "human_feedback": feedback}


def node_edit_review(state):
    """Revise the proposed review using the reviewer's requested changes."""
    a = state["analysis"]
    llm = get_llm().with_structured_output(PRAnalysis)
    revised = llm.invoke([
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
    return {"analysis": revised}


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


def _post(state, label: str) -> str:
    post_review_comment(state["pr_url"], _render_comment_body(state))
    console.print(f"  [green]OK[/green] posted comment to {state['pr_url']}")
    return label


def node_commit(state):
    console.print("[cyan]> commit[/cyan]")
    # Escalated reviews get a final human approval after synthesis.
    if state.get("human_choice") == "approve":
        label = "committed_after_escalation" if state.get("escalation_answers") is not None else "committed"
        return {"final_action": _post(state, label)}
    if state.get("human_choice") == "edit":
        label = "committed_edited_after_escalation" if state.get("escalation_answers") is not None else "committed_edited"
        return {"final_action": _post(state, label)}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]> auto_approve[/cyan]  [dim]high confidence - posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("edit_review", node_edit_review), ("commit", node_commit),
        ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_conditional_edges(
        "human_approval",
        lambda state: "edit_review" if state["human_choice"] == "edit" else "commit",
        {"edit_review": "edit_review", "commit": "commit"},
    )
    g.add_edge("edit_review", "commit")
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "human_approval")
    g.add_edge("commit", END)
    return g.compile(checkpointer=MemorySaver(serde=checkpoint_serializer()))


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        feedback = console.input("Feedback: ").strip() if choice != "approve" else ""
        while choice == "edit" and not feedback:
            feedback = console.input("Describe the requested edits: ").strip()
        return {"choice": choice, "feedback": feedback}
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        answers = {}
        for question in payload["questions"]:
            answer = ""
            while not answer:
                answer = console.input(f"Q: {question}\nA: ").strip()
            answers[question] = answer
        return answers
    raise ValueError(kind)


def main():
    load_dotenv()
    p = argparse.ArgumentParser(); p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
