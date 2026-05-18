"""Runtime helpers for the didactic LangGraph demo."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from uuid import uuid4

from news_system_demo.models import RunArtifactPaths, RuntimePaths


def build_runtime_paths(root_dir: Path) -> RuntimePaths:
    """Build the filesystem layout used by the demo runtime."""

    return RuntimePaths(
        root_dir=root_dir,
        corpus_path=root_dir / "corpus" / "news_corpus.json",
        runs_dir=root_dir / "runs",
        env_path=root_dir.parent / ".env",
    )


def ensure_runtime_directories(paths: RuntimePaths) -> None:
    """Create the directories required by the demo runtime."""

    paths.runs_dir.mkdir(parents=True, exist_ok=True)


def build_thread_id(topic: str) -> str:
    """Build a readable thread id from the topic and a short random suffix."""

    slug = "-".join(part for part in topic.lower().split() if part)[:40] or "demo-run"
    return f"{slug}-{uuid4().hex[:8]}"


def create_run_artifacts(paths: RuntimePaths, thread_id: str) -> RunArtifactPaths:
    """Create the per-run directory and artifact paths for one execution."""

    run_dir = paths.runs_dir / thread_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifactPaths(
        run_dir=str(run_dir),
        report_md=str(run_dir / "report.md"),
        run_log=str(run_dir / "run.log"),
    )


class RunLogger:
    """Human-readable plain-text logger for one demo run."""

    def __init__(self, artifacts: RunArtifactPaths, *, topic: str, truncate_existing: bool = True) -> None:
        """Initialize the logger with per-run artifact paths."""

        self.artifacts = artifacts
        self._step = 0
        if truncate_existing:
            Path(self.artifacts.run_log).write_text(
                f"# Run: {Path(self.artifacts.run_dir).name}\nTopic: {topic}\n\n",
                encoding="utf-8",
            )

    def step(self, name: str, message: str, details: Iterable[str] | None = None) -> None:
        """Append one readable step block to the run log and console."""

        self._step += 1
        lines = [f"## {self._step}. {name}", message]
        if details:
            lines.extend(f"- {detail}" for detail in details if detail)
        block = "\n".join(lines).rstrip() + "\n\n"
        print(block.rstrip())
        with Path(self.artifacts.run_log).open("a", encoding="utf-8") as handle:
            handle.write(block)

    def error(self, name: str, message: str) -> None:
        """Append an error block to the run log."""

        self.step(name, f"ERROR: {message}")
