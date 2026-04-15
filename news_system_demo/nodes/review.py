"""Review node for the didactic demo."""

from __future__ import annotations

import asyncio

from news_system_demo.llm import DemoLlmClientProtocol
from news_system_demo.models import DemoHandoff, DemoReviewPayload, DemoState, DemoWorkspace
from news_system_demo.nodes.shared import append_handoff, build_review_prompt
from news_system_demo.runtime import DemoTracer


def review_node(
    state: DemoState,
    *,
    llm_client: DemoLlmClientProtocol,
    tracer: DemoTracer,
) -> DemoState:
    """Review the draft and decide whether the graph should loop once."""

    draft = state["draft"]
    verifications = state.get("verifications", [])
    workspace = DemoWorkspace.model_validate(state["workspace"])
    revision_count = state.get("revision_count", 0)
    tracer.node(
        "review",
        "start",
        "Revisando el borrador para decidir si la organización debe volver a escribir.",
        {"revision_count": revision_count, "verifications": len(verifications)},
    )
    response = asyncio.run(
        llm_client.complete_json(
            system_prompt=(
                "Eres un revisor didáctico. Devuelve JSON con approved, feedback_summary, "
                "revision_instructions y strengths. Evalúa claridad, trazabilidad y ajuste "
                "a las cautelas del verificador."
            ),
            user_prompt=build_review_prompt(
                state["topic"],
                draft,
                verifications,
                revision_count,
            ),
            response_model=DemoReviewPayload,
            temperature=0.1,
        )
    )
    force_first_revision = workspace.force_first_revision and revision_count < 1
    needs_revision = force_first_revision or (
        not response.approved and revision_count < workspace.max_revisions
    )
    next_revision_count = revision_count + 1 if needs_revision else revision_count
    review_feedback = {
        "approved": not needs_revision,
        "needs_revision": needs_revision,
        "feedback_summary": response.feedback_summary,
        "revision_instructions": response.revision_instructions,
        "strengths": response.strengths,
        "forced_for_demo": force_first_revision,
    }
    next_target = "write" if needs_revision else "render"
    next_handoffs = append_handoff(
        state,
        tracer,
        DemoHandoff(
            from_agent="review",
            to_agent=next_target,
            purpose=(
                "Solicitar una nueva redacción."
                if needs_revision
                else "Aprobar el borrador y permitir su salida final."
            ),
            inputs_used=["draft", "verifications"],
            outputs_written=["review_feedback", "needs_revision", "revision_count"],
            summary=(
                "Review devuelve feedback al writer para una iteración adicional."
                if needs_revision
                else "Review aprueba el borrador y habilita la salida final."
            ),
        ),
    )
    tracer.node(
        "review",
        "end",
        "Revisión completada. La siguiente arista depende de la decisión del revisor.",
        {
            "needs_revision": needs_revision,
            "revision_count": next_revision_count,
            "forced_for_demo": force_first_revision,
        },
    )
    return {
        "review_feedback": review_feedback,
        "needs_revision": needs_revision,
        "revision_count": next_revision_count,
        "handoffs": next_handoffs,
    }


def route_after_review(state: DemoState, *, tracer: DemoTracer) -> str:
    """Choose the next node after review and trace the taken edge."""

    if state.get("needs_revision", False):
        tracer.edge(
            "review",
            "write",
            "Arista condicional tomada: el revisor pide una única reescritura visible.",
            {"revision_count": state.get("revision_count", 0)},
        )
        return "write"
    tracer.edge(
        "review",
        "render",
        "Arista condicional tomada: el revisor aprueba el borrador.",
        {"revision_count": state.get("revision_count", 0)},
    )
    return "render"
