"""Workspace node for the didactic demo."""

from __future__ import annotations

from news_system_demo.models import DemoHandoff, DemoRunArtifactPaths, DemoState
from news_system_demo.nodes.shared import append_handoff, load_demo_workspace, summarize_state
from news_system_demo.runtime import DemoTracer


def load_workspace_node(
    state: DemoState,
    *,
    artifacts: DemoRunArtifactPaths,
    tracer: DemoTracer,
) -> DemoState:
    """Load the tiny demo workspace and explain what enters shared state."""

    workspace = load_demo_workspace()
    tracer.node(
        "load_workspace",
        "start",
        "Inicializando el workspace mínimo de la demo.",
        {"topic": state["topic"]},
    )
    next_state: DemoState = {
        "workspace": workspace.model_dump(mode="json"),
        "artifacts": artifacts.model_dump(mode="json"),
        "handoffs": [],
        "revision_count": 0,
        "needs_revision": False,
    }
    next_handoffs = append_handoff(
        {**state, **next_state},
        tracer,
        DemoHandoff(
            from_agent="load_workspace",
            to_agent="research",
            purpose="Preparar el contexto común antes de leer noticias.",
            inputs_used=["topic"],
            outputs_written=["workspace", "artifacts", "revision_count"],
            summary="Workspace cargado y contexto compartido preparado para research.",
        ),
    )
    next_state["handoffs"] = next_handoffs
    tracer.node(
        "load_workspace",
        "end",
        "Workspace cargado. La organización ya conoce el tema, límites y política de revisión.",
        summarize_state({**state, **next_state}),
    )
    tracer.edge(
        "load_workspace",
        "research",
        "Transición lineal: con el workspace listo, la organización pasa a recopilar evidencia.",
    )
    return next_state
