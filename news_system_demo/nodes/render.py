"""Render node for the didactic demo."""

from __future__ import annotations

from pathlib import Path

from news_system_demo.models import DemoRunArtifactPaths, DemoState
from news_system_demo.nodes.shared import render_markdown_report
from news_system_demo.runtime import DemoTracer


def render_node(
    state: DemoState,
    *,
    artifacts: DemoRunArtifactPaths,
    tracer: DemoTracer,
) -> DemoState:
    """Render the final Markdown report and point to generated artifacts."""

    tracer.node(
        "render",
        "start",
        "Persistiendo el informe final y cerrando la ejecución observable.",
        {"run_dir": artifacts.run_dir},
    )
    report_md = render_markdown_report(state)
    Path(artifacts.report_md).write_text(report_md, encoding="utf-8")
    tracer.node(
        "render",
        "end",
        "Render completado. Los artefactos y la traza ya están fuera del grafo.",
        {
            "report_md": artifacts.report_md,
            "events_jsonl": artifacts.events_jsonl,
        },
    )
    tracer.edge(
        "render",
        "END",
        "La organización termina tras persistir el informe y la traza completa.",
    )
    return {"final_report": report_md}
