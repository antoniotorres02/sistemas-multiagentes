"""Workspace node for the didactic demo."""

from __future__ import annotations

from news_system_demo.models import State
from news_system_demo.runtime import RunLogger


def load_workspace_node(
    state: State,
    *,
    logger: RunLogger,
) -> State:
    """Initialize the small shared state used by the graph."""

    logger.step(
        "load_workspace",
        "Se prepara la ejecución y se inicializa el estado mínimo.",
        [f"Tema: {state['topic']}"],
    )
    return {"revision_count": 0, "needs_revision": False}
