"""Research node for the didactic demo."""

from __future__ import annotations

from collections.abc import Sequence

from news_system_demo.models import CorpusItem, State
from news_system_demo.nodes.shared import load_workspace, normalize_item, topic_matches_item
from news_system_demo.runtime import RunLogger


def research_node(
    state: State,
    *,
    corpus: Sequence[CorpusItem],
    logger: RunLogger,
) -> State:
    """Select local corpus items that match the user topic."""

    workspace = load_workspace()
    matched_items = [item for item in corpus if topic_matches_item(state["topic"], item)]
    selected_items = [normalize_item(item) for item in matched_items[: workspace.max_research_items]]
    logger.step(
        "research",
        "Se buscan noticias relevantes dentro del corpus local.",
        [
            f"Items encontrados: {len(selected_items)}",
            *[f"{item['source_name']}: {item['title']}" for item in selected_items],
        ],
    )
    return {"selected_items": selected_items}
