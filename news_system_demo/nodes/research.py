"""Research node for the didactic demo."""

from __future__ import annotations

from collections.abc import Sequence

from news_system_demo.models import CorpusItem, State, StateUpdate
from news_system_demo.nodes.shared import (
    MIN_MATCH_SCORE,
    load_workspace,
    normalize_item,
    score_item_for_topic,
)
from news_system_demo.runtime import RunLogger


def research_node(
    state: State,
    *,
    corpus: Sequence[CorpusItem],
    logger: RunLogger,
) -> StateUpdate:
    """Rank local corpus items that match the user topic."""

    workspace = load_workspace()
    ranked_items: list[tuple[int, list[str], CorpusItem]] = []
    for item in corpus:
        score, reasons = score_item_for_topic(state["topic"], item)
        if score >= MIN_MATCH_SCORE:
            ranked_items.append((score, reasons, item))
    ranked_items.sort(key=lambda candidate: (candidate[0], candidate[2].published_at), reverse=True)

    selected_items = [
        normalize_item(item, match_score=score, match_reasons=reasons)
        for score, reasons, item in ranked_items[: workspace.max_research_items]
    ]
    logger.step(
        "research",
        "Se simula una búsqueda editorial sobre el corpus local y se ordenan los candidatos.",
        [
            f"Candidatos evaluados: {len(corpus)}",
            f"Candidatos relevantes: {len(ranked_items)}",
            *[
                f"{item['match_score']} pts | {item['source_name']}: {item['title']} "
                f"({'; '.join(item['match_reasons'])})"
                for item in selected_items
            ],
        ],
    )
    return {"selected_items": selected_items}
