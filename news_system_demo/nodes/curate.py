"""Curate node for the didactic demo."""

from __future__ import annotations

from news_system_demo.models import State
from news_system_demo.nodes.shared import curate_items, load_workspace
from news_system_demo.runtime import RunLogger


def curate_node(
    state: State,
    *,
    logger: RunLogger,
) -> State:
    """Keep a compact evidence selection before verification."""

    workspace = load_workspace()
    selected_items = curate_items(
        state.get("selected_items", []),
        limit=workspace.max_story_briefs,
    )
    logger.step(
        "curate",
        "Se reduce la evidencia a una selección corta para redactar una noticia legible.",
        [
            f"Items mantenidos: {len(selected_items)}",
            *[f"{item['semantic_topic']}: {item['title']}" for item in selected_items],
        ],
    )
    return {"selected_items": selected_items}
