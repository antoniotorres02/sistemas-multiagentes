"""Write node for the didactic demo."""

from __future__ import annotations

import asyncio

from news_system_demo.llm import DemoLlmClientProtocol
from news_system_demo.models import DemoDraftPayload, DemoHandoff, DemoState
from news_system_demo.nodes.shared import append_handoff, build_no_evidence_draft, build_write_prompt
from news_system_demo.runtime import DemoTracer


def write_node(
    state: DemoState,
    *,
    llm_client: DemoLlmClientProtocol,
    tracer: DemoTracer,
) -> DemoState:
    """Draft a concise didactic report from verified stories."""

    story_briefs = state.get("story_briefs", [])
    verifications = state.get("verifications", [])
    review_feedback = state.get("review_feedback")
    revision_count = state.get("revision_count", 0)
    tracer.node(
        "write",
        "start",
        "Redactando un borrador corto y trazable a partir del estado verificado.",
        {"verified_stories": len(verifications), "revision_count": revision_count},
    )
    if not verifications:
        draft = build_no_evidence_draft(state["topic"])
    else:
        response = asyncio.run(
            llm_client.complete_json(
                system_prompt=(
                    "Eres un agente escritor didáctico. Devuelve JSON con headline, "
                    "summary, bullet_points y closing_note. Resume el flujo con "
                    "tono claro, menciona cautelas y respeta las instrucciones del revisor."
                ),
                user_prompt=build_write_prompt(
                    state["topic"],
                    story_briefs[: len(verifications)],
                    verifications,
                    review_feedback,
                    revision_count,
                ),
                response_model=DemoDraftPayload,
                temperature=0.2,
            )
        )
        draft = response.model_dump(mode="json")
    next_handoffs = append_handoff(
        state,
        tracer,
        DemoHandoff(
            from_agent="write",
            to_agent="review",
            purpose="Enviar un borrador legible para control de calidad.",
            inputs_used=["verifications", "review_feedback"],
            outputs_written=["draft"],
            summary="Write entrega un borrador que el revisor puede aprobar o devolver.",
        ),
    )
    tracer.node(
        "write",
        "end",
        "Borrador completado. La siguiente transición depende del revisor.",
        {"headline": draft["headline"], "bullet_points": len(draft["bullet_points"])},
    )
    tracer.edge(
        "write",
        "review",
        "El borrador sale del writer y pasa al control de calidad.",
    )
    return {"draft": draft, "handoffs": next_handoffs, "needs_revision": False}
