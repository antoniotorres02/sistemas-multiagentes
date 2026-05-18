"""CLI for the didactic LangGraph demo."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer

from news_system_demo.runtime import (
    RunLogger,
    build_runtime_paths,
    build_thread_id,
    create_run_artifacts,
    ensure_runtime_directories,
)
from news_system_demo.workflow import build_default_llm_client, build_demo_graph

app = typer.Typer(add_completion=False, no_args_is_help=True)
ROOT_DIR = Path(__file__).resolve().parent


async def _run_workflow(topic: str, thread_id: str) -> tuple[dict[str, Any], Any]:
    """Execute the workflow and persist report/log artifacts."""

    paths = build_runtime_paths(ROOT_DIR)
    ensure_runtime_directories(paths)
    artifacts = create_run_artifacts(paths, thread_id)
    logger = RunLogger(artifacts, topic=topic)
    llm_client = build_default_llm_client(paths.env_path)
    graph = build_demo_graph(
        corpus_path=paths.corpus_path,
        artifacts=artifacts,
        logger=logger,
        llm_client=llm_client,
    )
    final_state = await graph.ainvoke({"topic": topic, "thread_id": thread_id})
    return final_state, artifacts


@app.command("run")
def run_demo(
    topic: str = typer.Option(..., help="Tema a analizar en la demo."),
    thread_id: str | None = typer.Option(None, help="Thread id fijo para reabrir la misma ejecución."),
) -> None:
    """Run the workflow and persist a clean report plus a readable log."""

    effective_thread_id = thread_id or build_thread_id(topic)
    try:
        final_state, artifacts = asyncio.run(_run_workflow(topic, effective_thread_id))
        typer.echo("")
        typer.echo("=== news_system_demo COMPLETADA ===")
        typer.echo(f"thread_id: {effective_thread_id}")
        typer.echo(f"report_md: {artifacts.report_md}")
        typer.echo(f"run_log: {artifacts.run_log}")
        typer.echo("")
        typer.echo("Resumen:")
        typer.echo(
            json.dumps(
                {
                    "selected_items": len(final_state.get("selected_items", [])),
                    "revision_count": final_state.get("revision_count", 0),
                    "report_path": final_state.get("report_path"),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    except Exception as exc:
        paths = build_runtime_paths(ROOT_DIR)
        log_path = paths.runs_dir / effective_thread_id / "run.log"
        typer.echo("", err=True)
        typer.echo("=== news_system_demo FALLIDA ===", err=True)
        typer.echo(f"thread_id: {effective_thread_id}", err=True)
        typer.echo(f"error: {exc}", err=True)
        if log_path.exists():
            typer.echo(f"run_log: {log_path}", err=True)
        raise typer.Exit(code=1) from exc


def main() -> None:
    """Run the Typer CLI for the didactic demo."""

    app()
