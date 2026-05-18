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
    editorial_roles = ["pieza principal", "contexto", "contraste"]
    for index, item in enumerate(selected_items):
        item["editorial_role"] = editorial_roles[min(index, len(editorial_roles) - 1)]
    logger.step(
        "curate",
        "Se elige una mezcla corta de piezas para redactar una noticia legible.",
        [
            f"Items mantenidos: {len(selected_items)}",
            *[
                f"{item['editorial_role']}: {item['source_name']} | {item['title']} "
                f"[{item.get('story_angle', 'contexto')}]"
                for item in selected_items
            ],
        ],
    )
    return {"selected_items": selected_items}
