"""Runtime helpers for the didactic LangGraph demo."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.types import StateSnapshot

from news_system_demo.models import DemoHandoff, DemoRunArtifactPaths, DemoRuntimePaths, DemoTraceEvent


def build_runtime_paths(root_dir: Path) -> DemoRuntimePaths:
    """Build the filesystem layout used by the demo runtime."""

    return DemoRuntimePaths(
        root_dir=root_dir,
        corpus_path=root_dir / "corpus" / "news_corpus.json",
        data_dir=root_dir / "data",
        runs_dir=root_dir / "runs",
        checkpoint_db_path=root_dir / "data" / "checkpoints.sqlite3",
        env_path=root_dir.parent / ".env",
    )


def ensure_runtime_directories(paths: DemoRuntimePaths) -> None:
    """Create the directories required by the demo runtime."""

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)


def build_thread_id(topic: str) -> str:
    """Build a readable thread id from the topic and a short random suffix."""

    slug = "-".join(part for part in topic.lower().split() if part)[:40] or "demo-run"
    return f"{slug}-{uuid4().hex[:8]}"


def create_run_artifacts(paths: DemoRuntimePaths, thread_id: str) -> DemoRunArtifactPaths:
    """Create the per-run directory and artifact paths for one execution."""

    run_dir = paths.runs_dir / thread_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return DemoRunArtifactPaths(
        run_dir=str(run_dir),
        events_jsonl=str(run_dir / "events.jsonl"),
        graph_mermaid=str(run_dir / "graph.mmd"),
        report_md=str(run_dir / "report.md"),
        state_history_json=str(run_dir / "state_history.json"),
    )


class DemoTracer:
    """Console and JSONL tracer used by the demo workflow."""

    def __init__(self, artifacts: DemoRunArtifactPaths, *, truncate_existing: bool = True) -> None:
        """Initialize the tracer with per-run artifact paths."""

        self.artifacts = artifacts
        if truncate_existing:
            Path(self.artifacts.events_jsonl).write_text("", encoding="utf-8")

    def emit(
        self,
        event_kind: str,
        agent: str,
        phase: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit one trace event to console and JSONL."""

        event = DemoTraceEvent(
            event_kind=event_kind,  # type: ignore[arg-type]
            agent=agent,
            phase=phase,
            message=message,
            payload=payload or {},
        )
        timestamp = event.ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp} UTC] [{event.event_kind}] [{agent}] [{phase}] {message}")
        if event.payload:
            print(f"  payload={json.dumps(event.payload, ensure_ascii=False, sort_keys=True)}")
        with Path(self.artifacts.events_jsonl).open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")

    def node(
        self,
        agent: str,
        phase: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit one node-level trace event."""

        self.emit("node", agent, phase, message, payload)

    def edge(
        self,
        source: str,
        target: str,
        reason: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit one edge-transition trace event."""

        merged_payload = {"source": source, "target": target, **(payload or {})}
        self.emit("edge", source, "transition", reason, merged_payload)

    def handoff(self, handoff: DemoHandoff) -> None:
        """Emit one explicit inter-agent handoff."""

        self.emit(
            "handoff",
            handoff.from_agent,
            "handoff",
            handoff.summary,
            handoff.model_dump(mode="json"),
        )


def write_graph_mermaid(artifacts: DemoRunArtifactPaths) -> None:
    """Persist a Mermaid representation of the demo graph."""

    mermaid = """graph TD
    START([START]) --> load_workspace
    load_workspace --> research
    research --> curate
    curate --> verify
    verify --> write
    write --> review
    review -- needs_revision --> write
    review -- approved --> render
    render --> END([END])
    """
    Path(artifacts.graph_mermaid).write_text(mermaid, encoding="utf-8")


def serialize_state_history(snapshots: list[StateSnapshot]) -> list[dict[str, Any]]:
    """Convert LangGraph state snapshots into JSON-safe dictionaries."""

    serialized: list[dict[str, Any]] = []
    for snapshot in snapshots:
        serialized.append(
            {
                "created_at": snapshot.created_at,
                "metadata": snapshot.metadata,
                "next": list(snapshot.next),
                "config": snapshot.config,
                "parent_config": snapshot.parent_config,
                "values": snapshot.values,
            }
        )
    return serialized


def write_state_history(
    artifacts: DemoRunArtifactPaths,
    snapshots: list[StateSnapshot],
) -> None:
    """Persist the checkpoint-backed state history for a run."""

    Path(artifacts.state_history_json).write_text(
        json.dumps(serialize_state_history(snapshots), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
