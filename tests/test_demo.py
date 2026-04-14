"""Tests for the didactic LangGraph demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from typer.testing import CliRunner

from DEMO.cli import app
from DEMO.models import (
    DemoDraftPayload,
    DemoReviewPayload,
    DemoVerificationPayload,
    DemoRuntimePaths,
)
from DEMO.runtime import (
    DemoTracer,
    create_run_artifacts,
    write_graph_mermaid,
    write_state_history,
)
from DEMO.workflow import build_demo_graph


class FakeDemoLlmClient:
    """Fake LLM used by demo tests."""

    def __init__(self) -> None:
        """Track reviewer calls so the workflow loops once and then exits."""

        self.review_calls = 0

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> Any:
        """Return deterministic structured outputs for verify and write."""

        del system_prompt, user_prompt, temperature
        if response_model is DemoVerificationPayload:
            return DemoVerificationPayload.model_validate(
                {
                    "conclusion": "Las fuentes describen la misma historia de forma consistente.",
                    "claims": [
                        {
                            "text": "Las fuentes comparten el mismo eje temático.",
                            "supporting_points": ["Coinciden en tema y foco regulatorio."],
                        }
                    ],
                    "caution_notes": ["La demo usa un corpus local pequeño."],
                }
            )
        if response_model is DemoReviewPayload:
            self.review_calls += 1
            if self.review_calls == 1:
                return DemoReviewPayload.model_validate(
                    {
                        "approved": False,
                        "feedback_summary": (
                            "Hace falta una revisión visible para enseñar "
                            "la realimentación."
                        ),
                        "revision_instructions": [
                            "Haz más explícita la trazabilidad entre verificación y redacción.",
                            "Menciona que el bucle es deliberado y didáctico.",
                        ],
                        "strengths": ["La estructura del borrador ya es clara."],
                    }
                )
            return DemoReviewPayload.model_validate(
                {
                    "approved": True,
                    "feedback_summary": "La segunda versión ya es suficientemente clara.",
                    "revision_instructions": [],
                    "strengths": [
                        "La trazabilidad es explícita.",
                        "El flujo multiagente se entiende con una sola lectura.",
                    ],
                }
            )
        return DemoDraftPayload.model_validate(
            {
                "headline": "Resumen didáctico generado",
                "summary": (
                    "El flujo recorre research, curate, verify, write y review "
                    "para dejar visible la coordinación."
                ),
                "bullet_points": [
                    "El estado compartido acumula datos entre agentes.",
                    "Los checkpoints y handoffs permiten inspección posterior.",
                ],
                "closing_note": (
                    "La DEMO prioriza claridad arquitectónica sobre complejidad "
                    "de dominio."
                ),
            }
        )


def test_demo_workflow_runs_and_persists_artifacts(tmp_path: Path) -> None:
    """Run the demo graph end to end with a fake LLM and persist artifacts."""

    root_dir = tmp_path / "DEMO"
    corpus_dir = root_dir / "corpus"
    data_dir = root_dir / "data"
    runs_dir = root_dir / "runs"
    corpus_dir.mkdir(parents=True)
    data_dir.mkdir()
    runs_dir.mkdir()
    source_corpus = Path(__file__).resolve().parents[1] / "DEMO" / "corpus" / "news_corpus.json"
    corpus_path = corpus_dir / "news_corpus.json"
    corpus_path.write_text(source_corpus.read_text(encoding="utf-8"), encoding="utf-8")

    runtime_paths = DemoRuntimePaths(
        root_dir=root_dir,
        corpus_path=corpus_path,
        data_dir=data_dir,
        runs_dir=runs_dir,
        checkpoint_db_path=data_dir / "checkpoints.sqlite3",
        env_path=tmp_path / ".env",
    )
    artifacts = create_run_artifacts(runtime_paths, "demo-thread")
    write_graph_mermaid(artifacts)
    tracer = DemoTracer(artifacts)

    with SqliteSaver.from_conn_string(str(data_dir / "checkpoints.sqlite3")) as saver:
        graph = build_demo_graph(
            corpus_path=corpus_path,
            artifacts=artifacts,
            tracer=tracer,
            llm_client=FakeDemoLlmClient(),
            checkpointer=saver,
        )
        config = {"configurable": {"thread_id": "demo-thread"}}
        final_state = graph.invoke(
            {"topic": "ai regulation europe", "thread_id": "demo-thread"},
            config=config,
        )
        snapshots = list(graph.get_state_history(config))
        write_state_history(artifacts, snapshots)

    assert "final_report" in final_state
    assert len(final_state["story_briefs"]) >= 1
    assert len(final_state["verifications"]) >= 1
    assert final_state["revision_count"] == 1
    assert len(final_state["handoffs"]) >= 6
    assert Path(artifacts.report_md).exists()
    assert Path(artifacts.events_jsonl).exists()
    assert Path(artifacts.state_history_json).exists()
    assert "Resumen didáctico generado" in Path(artifacts.report_md).read_text(encoding="utf-8")
    assert "### Handoffs" in Path(artifacts.report_md).read_text(encoding="utf-8")


def test_demo_replay_cli_reads_persisted_state_history(tmp_path: Path, monkeypatch: Any) -> None:
    """Replay a persisted demo history through the CLI."""

    runner = CliRunner()
    demo_root = tmp_path / "DEMO"
    (demo_root / "runs" / "demo-thread").mkdir(parents=True)
    history_path = demo_root / "runs" / "demo-thread" / "state_history.json"
    history_path.write_text(
        json.dumps(
            [
                {
                    "created_at": "2026-03-25T20:00:00+00:00",
                    "metadata": {"step": 0},
                    "values": {"topic": "ai regulation europe"},
                },
                {
                    "created_at": "2026-03-25T20:00:01+00:00",
                    "metadata": {"step": 1},
                    "values": {
                        "topic": "ai regulation europe",
                        "story_briefs": [{"story_id": "ai-reg"}],
                    },
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("DEMO.cli.ROOT_DIR", demo_root)

    result = runner.invoke(app, ["replay", "--thread-id", "demo-thread", "--checkpoint-index", "1"])

    assert result.exit_code == 0
    assert "demo-thread" in result.stdout
    assert "\"story_briefs\"" in result.stdout


def test_demo_show_trace_cli_reads_persisted_trace(tmp_path: Path, monkeypatch: Any) -> None:
    """Show the persisted trace with edge and handoff events."""

    runner = CliRunner()
    demo_root = tmp_path / "DEMO"
    (demo_root / "runs" / "demo-thread").mkdir(parents=True)
    events_path = demo_root / "runs" / "demo-thread" / "events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-03-25T20:00:00+00:00",
                        "event_kind": "node",
                        "agent": "research",
                        "phase": "start",
                        "message": "Inicio research",
                        "payload": {"topic": "ai regulation europe"},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "ts": "2026-03-25T20:00:01+00:00",
                        "event_kind": "edge",
                        "agent": "review",
                        "phase": "transition",
                        "message": "Arista condicional tomada",
                        "payload": {"source": "review", "target": "write"},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "ts": "2026-03-25T20:00:02+00:00",
                        "event_kind": "handoff",
                        "agent": "review",
                        "phase": "handoff",
                        "message": "Review devuelve feedback",
                        "payload": {"from_agent": "review", "to_agent": "write"},
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("DEMO.cli.ROOT_DIR", demo_root)

    result = runner.invoke(app, ["show-trace", "--thread-id", "demo-thread"])

    assert result.exit_code == 0
    assert "[edge]" in result.stdout
    assert "review/transition" in result.stdout
    assert "\"target\": \"write\"" in result.stdout
    assert "Review devuelve feedback" in result.stdout
