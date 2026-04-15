"""Curate node for the didactic demo."""

from __future__ import annotations

from news_system_demo.models import DemoHandoff, DemoState, DemoWorkspace
from news_system_demo.nodes.shared import append_handoff, build_story_briefs
from news_system_demo.runtime import DemoTracer


def curate_node(
    state: DemoState,
    *,
    tracer: DemoTracer,
) -> DemoState:
    """Group research items into a smaller set of didactic story briefs."""

    workspace = DemoWorkspace.model_validate(state["workspace"])
    raw_items = state.get("raw_items", [])
    tracer.node(
        "curate",
        "start",
        "Agrupando artículos en historias observables para reducir complejidad narrativa.",
        {"raw_items": len(raw_items)},
    )
    story_briefs = build_story_briefs(raw_items)[: workspace.max_story_briefs]
    next_handoffs = append_handoff(
        state,
        tracer,
        DemoHandoff(
            from_agent="curate",
            to_agent="verify",
            purpose="Pasar historias agrupadas y priorizadas al verificador.",
            inputs_used=["raw_items"],
            outputs_written=["story_briefs"],
            summary="Curate reduce el ruido y entrega historias agrupadas al verificador.",
        ),
    )
    tracer.node(
        "curate",
        "end",
        "Curación completada. El estado ya expresa historias, no artículos sueltos.",
        {
            "story_ids": [story["story_id"] for story in story_briefs],
            "story_sizes": [len(story["items"]) for story in story_briefs],
        },
    )
    tracer.edge(
        "curate",
        "verify",
        "Las historias agrupadas se transfieren al verificador para juicio semántico.",
    )
    return {"story_briefs": story_briefs, "handoffs": next_handoffs}
