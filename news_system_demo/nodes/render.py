"""Render node for the didactic demo."""

from __future__ import annotations

from pathlib import Path

from news_system_demo.models import RunArtifactPaths, State, StateUpdate
from news_system_demo.nodes.shared import render_markdown_report
from news_system_demo.runtime import RunLogger


def render_node(
    state: State,
    *,
    artifacts: RunArtifactPaths,
    logger: RunLogger,
) -> StateUpdate:
    """Render the final Markdown report."""

    report_md = render_markdown_report(state)
    Path(artifacts.report_md).write_text(report_md, encoding="utf-8")
    logger.step(
        "render",
        "Se guarda el report final como noticia limpia.",
        [f"report.md: {artifacts.report_md}"],
    )
    return {"report_path": artifacts.report_md}
