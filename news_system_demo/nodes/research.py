"""Research node for the didactic demo."""

from __future__ import annotations

from collections.abc import Sequence

from news_system_demo.models import DemoCorpusItem, DemoHandoff, DemoState, DemoWorkspace
from news_system_demo.nodes.shared import append_handoff, normalize_research_item, topic_matches_item
from news_system_demo.runtime import DemoTracer


def research_node(
    state: DemoState,
    *,
    corpus: Sequence[DemoCorpusItem],
    tracer: DemoTracer,
) -> DemoState:
    """Select and normalize local corpus items that match the user topic."""

    workspace = DemoWorkspace.model_validate(state["workspace"])
    tracer.node(
        "research",
        "start",
        "Seleccionando evidencia del corpus local y dejándola en un formato homogéneo.",
        {"topic": state["topic"], "corpus_items": len(corpus)},
    )
    matched_items = [item for item in corpus if topic_matches_item(state["topic"], item)]
    limited_items = matched_items[: workspace.max_research_items]
    raw_items = [normalize_research_item(item) for item in limited_items]
    next_handoffs = append_handoff(
        state,
        tracer,
        DemoHandoff(
            from_agent="research",
            to_agent="curate",
            purpose="Entregar evidencia comparable para agrupar historias.",
            inputs_used=["topic", "workspace"],
            outputs_written=["raw_items"],
            summary="Research deja artículos homogéneos y listos para curación.",
        ),
    )
    tracer.node(
        "research",
        "end",
        "Research completado. El estado ya contiene evidencia local normalizada.",
        {"raw_titles": [item["title"] for item in raw_items]},
    )
    tracer.edge(
        "research",
        "curate",
        "La evidencia ya es comparable; el siguiente agente puede agruparla por historia.",
    )
    return {"raw_items": raw_items, "handoffs": next_handoffs}
