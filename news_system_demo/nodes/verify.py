"""Verify node for the didactic demo."""

from __future__ import annotations

import asyncio
from typing import Any

from news_system_demo.llm import DemoLlmClientProtocol
from news_system_demo.models import DemoHandoff, DemoState, DemoVerificationPayload, DemoWorkspace
from news_system_demo.nodes.shared import append_handoff, build_verify_prompt
from news_system_demo.runtime import DemoTracer


def verify_node(
    state: DemoState,
    *,
    llm_client: DemoLlmClientProtocol,
    tracer: DemoTracer,
) -> DemoState:
    """Verify the most relevant story briefs with a real LLM call."""

    story_briefs = state.get("story_briefs", [])
    workspace = DemoWorkspace.model_validate(state["workspace"])
    tracer.node(
        "verify",
        "start",
        "Verificando historias con LLM real para añadir conclusiones y cautelas explícitas.",
        {
            "story_briefs": len(story_briefs),
            "max_written_stories": workspace.max_written_stories,
        },
    )
    verifications: list[dict[str, Any]] = []
    for story_brief in story_briefs[: workspace.max_written_stories]:
        response = asyncio.run(
            llm_client.complete_json(
                system_prompt=(
                    "Eres un agente verificador didáctico. Devuelve JSON con "
                    "conclusion, claims y caution_notes. Resume si varias fuentes "
                    "hablan de la misma historia y qué matices conviene no exagerar."
                ),
                user_prompt=build_verify_prompt(story_brief, state["topic"]),
                response_model=DemoVerificationPayload,
                temperature=0.1,
            )
        )
        verification = {
            "story_id": story_brief["story_id"],
            "conclusion": response.conclusion,
            "claims": response.model_dump(mode="json")["claims"],
            "caution_notes": response.caution_notes,
            "independent_source_count": story_brief["independent_source_count"],
            "confidence_score": min(
                0.45 + (0.2 * max(story_brief["independent_source_count"] - 1, 0)),
                0.95,
            ),
        }
        tracer.node(
            "verify",
            "story",
            "Historia verificada. El juicio del LLM queda externalizado en el estado.",
            {
                "story_id": verification["story_id"],
                "confidence_score": verification["confidence_score"],
            },
        )
        verifications.append(verification)
    next_handoffs = append_handoff(
        state,
        tracer,
        DemoHandoff(
            from_agent="verify",
            to_agent="write",
            purpose="Entregar conclusiones verificadas al redactor.",
            inputs_used=["story_briefs"],
            outputs_written=["verifications"],
            summary="Verify añade juicio y cautelas antes de la redacción.",
        ),
    )
    tracer.node(
        "verify",
        "end",
        "Verificación completada. El writer ya puede redactar con soporte explícito.",
        {"verified_stories": len(verifications)},
    )
    tracer.edge(
        "verify",
        "write",
        "Con la evidencia verificada, el writer puede sintetizar un borrador.",
    )
    return {"verifications": verifications, "handoffs": next_handoffs}
