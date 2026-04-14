"""CLI for the didactic LangGraph demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from langgraph.checkpoint.sqlite import SqliteSaver

from DEMO.models import DemoTraceEvent
from DEMO.runtime import (
    build_runtime_paths,
    build_thread_id,
    create_run_artifacts,
    ensure_runtime_directories,
    write_graph_mermaid,
    write_state_history,
)
from DEMO.workflow import build_default_llm_client, build_demo_graph

app = typer.Typer(add_completion=False, no_args_is_help=True)
ROOT_DIR = Path(__file__).resolve().parent


def _with_graph(topic: str, thread_id: str) -> tuple[Any, Any, Any]:
    """Build the demo graph, tracer and artifacts for one execution context."""

    paths = build_runtime_paths(ROOT_DIR)
    ensure_runtime_directories(paths)
    artifacts = create_run_artifacts(paths, thread_id)
    write_graph_mermaid(artifacts)
    from DEMO.runtime import DemoTracer

    tracer = DemoTracer(artifacts)
    llm_client = build_default_llm_client(paths.env_path)
    saver_context = SqliteSaver.from_conn_string(str(paths.checkpoint_db_path))
    saver = saver_context.__enter__()
    graph = build_demo_graph(
        corpus_path=paths.corpus_path,
        artifacts=artifacts,
        tracer=tracer,
        llm_client=llm_client,
        checkpointer=saver,
    )
    return graph, saver_context, artifacts


class _StubLlmClient:
    """Stub client used for read-only commands that never execute LLM nodes."""

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> Any:
        """Raise if a read-only command accidentally tries to execute the graph."""

        del system_prompt, user_prompt, response_model, temperature
        raise RuntimeError("The stub LLM client cannot execute workflow nodes.")


def _build_config(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    """Build the LangGraph runnable config for a demo thread and optional checkpoint."""

    configurable: dict[str, str] = {"thread_id": thread_id}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _read_trace_events(thread_id: str) -> list[DemoTraceEvent]:
    """Read persisted trace events for one run directory."""

    paths = build_runtime_paths(ROOT_DIR)
    events_path = paths.runs_dir / thread_id / "events.jsonl"
    if not events_path.exists():
        raise typer.BadParameter(f"No events.jsonl found for thread_id={thread_id}.")
    events: list[DemoTraceEvent] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(DemoTraceEvent.model_validate_json(line))
    return events


@app.command("run")
def run_demo(
    topic: str = typer.Option(..., help="Tema didáctico a analizar en la DEMO."),
    thread_id: str | None = typer.Option(None, help="Thread id fijo para demostrar persistencia."),
) -> None:
    """Run the didactic workflow and persist both artifacts and checkpoints."""

    effective_thread_id = thread_id or build_thread_id(topic)
    graph, saver_context, artifacts = _with_graph(topic, effective_thread_id)
    try:
        config = _build_config(effective_thread_id)
        final_state = graph.invoke(
            {"topic": topic, "thread_id": effective_thread_id},
            config=config,
        )
        snapshots = list(graph.get_state_history(config))
        write_state_history(artifacts, snapshots)
        typer.echo("")
        typer.echo("=== DEMO COMPLETADA ===")
        typer.echo(f"thread_id: {effective_thread_id}")
        typer.echo(f"report_md: {artifacts.report_md}")
        typer.echo(f"events_jsonl: {artifacts.events_jsonl}")
        typer.echo(f"graph_mermaid: {artifacts.graph_mermaid}")
        typer.echo(f"state_history_json: {artifacts.state_history_json}")
        typer.echo(f"checkpoint_db: {build_runtime_paths(ROOT_DIR).checkpoint_db_path}")
        typer.echo("")
        typer.echo("Resumen de la run:")
        typer.echo(
            json.dumps(
                {
                    "story_briefs": len(final_state.get("story_briefs", [])),
                    "verifications": len(final_state.get("verifications", [])),
                    "handoffs": len(final_state.get("handoffs", [])),
                    "revision_count": final_state.get("revision_count", 0),
                    "review_feedback": final_state.get("review_feedback", {}),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    finally:
        saver_context.__exit__(None, None, None)


@app.command("show-history")
def show_history(
    thread_id: str = typer.Option(..., help="Thread id persistido por una ejecución anterior."),
) -> None:
    """Print the checkpoint-backed history for one demo thread."""

    paths = build_runtime_paths(ROOT_DIR)
    ensure_runtime_directories(paths)
    saver_context = SqliteSaver.from_conn_string(str(paths.checkpoint_db_path))
    saver = saver_context.__enter__()
    try:
        from DEMO.runtime import DemoTracer

        artifacts = create_run_artifacts(paths, thread_id)
        tracer = DemoTracer(artifacts, truncate_existing=False)
        graph = build_demo_graph(
            corpus_path=paths.corpus_path,
            artifacts=artifacts,
            tracer=tracer,
            llm_client=_StubLlmClient(),
            checkpointer=saver,
        )
        history = list(graph.get_state_history(_build_config(thread_id)))
        typer.echo(f"Historial de checkpoints para thread_id={thread_id}")
        for index, snapshot in enumerate(history):
            checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
            typer.echo(
                f"[{index}] checkpoint_id={checkpoint_id} created_at={snapshot.created_at} "
                f"step={snapshot.metadata.get('step')} next={list(snapshot.next)}"
            )
            typer.echo(
                json.dumps(
                    {
                        "keys": sorted(snapshot.values.keys()),
                        "values_summary": {
                            key: len(value) if isinstance(value, list) else type(value).__name__
                            for key, value in snapshot.values.items()
                        },
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        saver_context.__exit__(None, None, None)


@app.command("show-state")
def show_state(
    thread_id: str = typer.Option(..., help="Thread id persistido por una ejecución anterior."),
    checkpoint_id: str | None = typer.Option(None, help="Checkpoint concreto a inspeccionar."),
) -> None:
    """Print the shared state for the latest or a selected checkpoint."""

    paths = build_runtime_paths(ROOT_DIR)
    ensure_runtime_directories(paths)
    saver_context = SqliteSaver.from_conn_string(str(paths.checkpoint_db_path))
    saver = saver_context.__enter__()
    try:
        from DEMO.runtime import DemoTracer

        artifacts = create_run_artifacts(paths, thread_id)
        tracer = DemoTracer(artifacts, truncate_existing=False)
        graph = build_demo_graph(
            corpus_path=paths.corpus_path,
            artifacts=artifacts,
            tracer=tracer,
            llm_client=_StubLlmClient(),
            checkpointer=saver,
        )
        snapshot = graph.get_state(_build_config(thread_id, checkpoint_id))
        typer.echo(
            json.dumps(
                {
                    "created_at": snapshot.created_at,
                    "checkpoint": snapshot.config,
                    "metadata": snapshot.metadata,
                    "next": list(snapshot.next),
                    "values": snapshot.values,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    finally:
        saver_context.__exit__(None, None, None)


@app.command("show-trace")
def show_trace(
    thread_id: str = typer.Option(..., help="Thread id persistido por una ejecución anterior."),
    event_kind: str | None = typer.Option(
        None,
        help="Filtra por tipo de evento: node, edge o handoff.",
    ),
) -> None:
    """Print the persisted trace showing nodes, edges and handoffs."""

    events = _read_trace_events(thread_id)
    typer.echo(f"Traza persistida para thread_id={thread_id}")
    for index, event in enumerate(events):
        if event_kind is not None and event.event_kind != event_kind:
            continue
        typer.echo(
            f"[{index}] {event.ts.isoformat()} [{event.event_kind}] "
            f"{event.agent}/{event.phase}: {event.message}"
        )
        if event.payload:
            typer.echo(json.dumps(event.payload, ensure_ascii=False, indent=2))


@app.command("replay")
def replay_history(
    thread_id: str = typer.Option(..., help="Thread id persistido por una ejecución anterior."),
    checkpoint_index: int = typer.Option(
        0,
        help="Índice del checkpoint desde el que quieres re-leer la ejecución.",
    ),
) -> None:
    """Replay the saved checkpoint history from one selected point forward."""

    paths = build_runtime_paths(ROOT_DIR)
    history_path = paths.runs_dir / thread_id / "state_history.json"
    if not history_path.exists():
        raise typer.BadParameter(f"No state history found for thread_id={thread_id}.")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if checkpoint_index < 0 or checkpoint_index >= len(history):
        raise typer.BadParameter("checkpoint_index is outside the available snapshot range.")
    typer.echo(
        f"Replaying checkpoint history from index {checkpoint_index} "
        f"for thread_id={thread_id}"
    )
    for index, snapshot in enumerate(history[checkpoint_index:], start=checkpoint_index):
        typer.echo(
            f"[{index}] created_at={snapshot['created_at']} "
            f"step={snapshot['metadata'].get('step')} "
            f"keys={sorted(snapshot['values'].keys())}"
        )
        typer.echo(json.dumps(snapshot["values"], ensure_ascii=False, indent=2))


def main() -> None:
    """Run the Typer CLI for the didactic demo."""

    app()
