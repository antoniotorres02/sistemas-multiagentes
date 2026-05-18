"""Tests for the didactic LangGraph demo."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from news_system_demo import llm as llm_module
from news_system_demo.cli import app
from news_system_demo.llm import OpenRouterClient
from news_system_demo.models import ArticlePayload, EvidencePayload, ReviewPayload, RuntimePaths
from news_system_demo.nodes.shared import curate_items, load_corpus, score_item_for_topic
from news_system_demo.runtime import RunLogger, create_run_artifacts
from news_system_demo.workflow import build_demo_graph


class FakeLlmClient:
    """Fake LLM used by tests to avoid network calls."""

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> Any:
        """Return deterministic structured responses."""

        del system_prompt, user_prompt, temperature
        if response_model is EvidencePayload:
            return EvidencePayload(evidence_note="La evidencia apunta a regulación europea con cautelas.")
        if response_model is ReviewPayload:
            return ReviewPayload(approved=True, note="La noticia es clara y suficiente.")
        if response_model is ArticlePayload:
            return ArticlePayload(
                article_text="\n".join(
                    [
                        "# La regulación europea de IA avanza con cautelas",
                        "",
                        "La evidencia local muestra que el debate regulatorio sigue activo.",
                        "",
                        "El texto destaca obligaciones, plazos y la necesidad de no exagerar el alcance.",
                        "",
                        "## Fuentes",
                        "- European Policy Daily",
                    ]
                )
            )
        raise AssertionError(f"Unexpected response model: {response_model}")


class RejectingReviewLlmClient(FakeLlmClient):
    """Fake LLM whose reviewer never approves the article."""

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> Any:
        """Reject only review responses."""

        if response_model is ReviewPayload:
            return ReviewPayload(approved=False, note="Necesita una segunda pasada editorial.")
        return await super().complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            temperature=temperature,
        )


class AlwaysFailLlmClient:
    """Fake LLM that always raises to validate fail-fast behavior."""

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        temperature: float = 0.2,
    ) -> Any:
        """Raise a deterministic error instead of returning structured output."""

        del system_prompt, user_prompt, response_model, temperature
        raise RuntimeError("forced llm failure")


class FakeStructuredModel:
    """Fake LangChain structured runnable used by OpenRouterClient tests."""

    def __init__(self, chat_model: "FakeChatOpenRouter", response_model: type[Any]) -> None:
        self.chat_model = chat_model
        self.response_model = response_model

    async def ainvoke(self, messages: list[tuple[str, str]]) -> Any:
        """Return a deterministic payload or raise the configured error."""

        self.chat_model.messages = messages
        if self.chat_model.error is not None:
            raise self.chat_model.error
        if self.response_model is EvidencePayload:
            return EvidencePayload(evidence_note="La evidencia es suficiente.")
        raise AssertionError(f"Unexpected response model: {self.response_model}")


class FakeChatOpenRouter:
    """Small test double for langchain_openrouter.ChatOpenRouter."""

    instances: list["FakeChatOpenRouter"] = []
    error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.structured_calls: list[dict[str, Any]] = []
        self.messages: list[tuple[str, str]] = []
        self.error = self.__class__.error
        self.__class__.instances.append(self)

    def with_structured_output(
        self,
        response_model: type[Any],
        *,
        method: str,
        strict: bool,
    ) -> FakeStructuredModel:
        """Capture structured-output options and return a fake runnable."""

        self.structured_calls.append(
            {"response_model": response_model, "method": method, "strict": strict}
        )
        return FakeStructuredModel(self, response_model)


def _build_demo_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a minimal demo filesystem tree for isolated tests."""

    root_dir = tmp_path / "news_system_demo"
    corpus_dir = root_dir / "corpus"
    runs_dir = root_dir / "runs"
    corpus_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)
    corpus_path = corpus_dir / "news_corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "item_id": "a",
                    "title": "EU AI Act timeline advances",
                    "summary": "EU institutions prepare phased AI obligations.",
                    "body": "The AI Act introduces obligations with phased implementation dates.",
                    "source_name": "European Policy Daily",
                    "url": "https://example.com/eu-ai-act",
                    "published_at": "2026-03-01T12:00:00Z",
                    "tags": ["ai", "regulation", "europe"],
                    "semantic_topic": "ai regulation",
                },
                {
                    "item_id": "b",
                    "title": "Hospitals test AI triage",
                    "summary": "Clinics run controlled pilots.",
                    "body": "Health systems evaluate AI triage with medical supervision.",
                    "source_name": "Health Systems Review",
                    "url": "https://example.com/ai-health",
                    "published_at": "2026-03-02T12:00:00Z",
                    "tags": ["ai", "health"],
                    "semantic_topic": "ai in health",
                },
            ]
        ),
        encoding="utf-8",
    )
    return root_dir, corpus_path, runs_dir


def test_workflow_writes_clean_report_and_plain_log(tmp_path: Path) -> None:
    """Run the graph end to end and persist only report.md plus run.log."""

    root_dir, corpus_path, runs_dir = _build_demo_root(tmp_path)
    runtime_paths = RuntimePaths(
        root_dir=root_dir,
        corpus_path=corpus_path,
        runs_dir=runs_dir,
        env_path=tmp_path / ".env",
    )
    artifacts = create_run_artifacts(runtime_paths, "demo-thread")
    logger = RunLogger(artifacts, topic="ai regulation europe")
    graph = build_demo_graph(
        corpus_path=corpus_path,
        artifacts=artifacts,
        logger=logger,
        llm_client=FakeLlmClient(),
    )

    final_state = asyncio.run(
        graph.ainvoke({"topic": "ai regulation europe", "thread_id": "demo-thread"})
    )

    report_text = Path(artifacts.report_md).read_text(encoding="utf-8")
    log_text = Path(artifacts.run_log).read_text(encoding="utf-8")
    assert final_state["report_path"] == artifacts.report_md
    assert "# La regulación europea de IA avanza con cautelas" in report_text
    assert "Handoffs" not in report_text
    assert "Revision" not in report_text
    assert "estado interno" not in report_text
    assert "## 1. load_workspace" in log_text
    assert "## 2. research" in log_text
    assert "## 6. review" in log_text
    assert "## 7. render" in log_text
    assert not Path(artifacts.run_dir, "events.jsonl").exists()
    assert not Path(artifacts.run_dir, "state_history.json").exists()


def test_research_ranking_prefers_relevant_editorial_candidates(tmp_path: Path) -> None:
    """Score the richer corpus by topic instead of taking the first boolean match."""

    _, corpus_path, _ = _build_demo_root(tmp_path)
    corpus = load_corpus(corpus_path)
    ranked = sorted(
        (
            (score, item.item_id, reasons)
            for item in corpus
            for score, reasons in [score_item_for_topic("ai regulation europe", item)]
            if score > 0
        ),
        reverse=True,
    )

    assert ranked[0][1] == "a"
    assert "tema directo: ai regulation" in ranked[0][2]


def test_curate_keeps_source_diversity_before_filling_limit() -> None:
    """Avoid keeping near-duplicates when a different source is available."""

    items = [
        {"title": "a", "semantic_topic": "ai regulation", "source_name": "Same Source", "match_score": 80},
        {"title": "b", "semantic_topic": "ai regulation", "source_name": "Same Source", "match_score": 75},
        {"title": "c", "semantic_topic": "ai regulation", "source_name": "Different Source", "match_score": 70},
    ]

    selected = curate_items(items, limit=2)

    assert [item["title"] for item in selected] == ["a", "c"]


def test_unrelated_topic_keeps_no_evidence_fallback(tmp_path: Path) -> None:
    """Do not invent a story when the topic has no corpus support."""

    root_dir, corpus_path, runs_dir = _build_demo_root(tmp_path)
    runtime_paths = RuntimePaths(
        root_dir=root_dir,
        corpus_path=corpus_path,
        runs_dir=runs_dir,
        env_path=tmp_path / ".env",
    )
    artifacts = create_run_artifacts(runtime_paths, "demo-thread")
    logger = RunLogger(artifacts, topic="volcanic tourism")
    graph = build_demo_graph(
        corpus_path=corpus_path,
        artifacts=artifacts,
        logger=logger,
        llm_client=FakeLlmClient(),
    )

    final_state = asyncio.run(graph.ainvoke({"topic": "volcanic tourism", "thread_id": "demo-thread"}))

    assert final_state["selected_items"] == []
    assert "No hay evidencia suficiente" in Path(artifacts.report_md).read_text(encoding="utf-8")


def test_review_limit_does_not_fake_approval(tmp_path: Path) -> None:
    """Finish by revision limit without pretending the reviewer approved."""

    root_dir, corpus_path, runs_dir = _build_demo_root(tmp_path)
    runtime_paths = RuntimePaths(
        root_dir=root_dir,
        corpus_path=corpus_path,
        runs_dir=runs_dir,
        env_path=tmp_path / ".env",
    )
    artifacts = create_run_artifacts(runtime_paths, "demo-thread")
    logger = RunLogger(artifacts, topic="ai regulation europe")
    graph = build_demo_graph(
        corpus_path=corpus_path,
        artifacts=artifacts,
        logger=logger,
        llm_client=RejectingReviewLlmClient(),
    )

    final_state = asyncio.run(
        graph.ainvoke({"topic": "ai regulation europe", "thread_id": "demo-thread"})
    )

    assert final_state["revision_count"] == 1
    assert final_state["needs_revision"] is False
    assert final_state["review_note"] == "Necesita una segunda pasada editorial."
    assert "límite de revisiones" in Path(artifacts.run_log).read_text(encoding="utf-8")


def test_openrouter_client_uses_langchain_structured_output(monkeypatch: Any) -> None:
    """Call ChatOpenRouter through LangChain structured output."""

    FakeChatOpenRouter.instances = []
    FakeChatOpenRouter.error = None
    monkeypatch.setattr(llm_module, "ChatOpenRouter", FakeChatOpenRouter)
    client = OpenRouterClient(api_key="test-key", model="test-model")

    response = asyncio.run(
        client.complete_json(
            system_prompt="system",
            user_prompt="user",
            response_model=EvidencePayload,
            temperature=0.2,
        )
    )

    assert response == EvidencePayload(evidence_note="La evidencia es suficiente.")
    assert len(FakeChatOpenRouter.instances) == 1
    fake_model = FakeChatOpenRouter.instances[0]
    assert fake_model.kwargs["api_key"] == "test-key"
    assert fake_model.kwargs["model"] == "test-model"
    assert fake_model.kwargs["temperature"] == 0.2
    assert fake_model.kwargs["timeout"] == 60_000
    assert fake_model.kwargs["max_retries"] == 2
    assert fake_model.kwargs["app_title"] == "news-system-demo"
    assert fake_model.kwargs["openrouter_provider"] == {"require_parameters": True}
    assert fake_model.structured_calls == [
        {"response_model": EvidencePayload, "method": "json_schema", "strict": True}
    ]
    assert fake_model.messages == [("system", "system"), ("human", "user")]


def test_openrouter_client_fails_fast_when_langchain_fails(monkeypatch: Any) -> None:
    """Raise an explicit error instead of returning a synthetic fallback payload."""

    FakeChatOpenRouter.instances = []
    FakeChatOpenRouter.error = ValueError("structured output failed")
    monkeypatch.setattr(llm_module, "ChatOpenRouter", FakeChatOpenRouter)
    client = OpenRouterClient(api_key="test-key", model="test-model")

    async def _run() -> None:
        await client.complete_json(
            system_prompt="system",
            user_prompt="user",
            response_model=EvidencePayload,
        )

    try:
        asyncio.run(_run())
    except RuntimeError as exc:
        assert "OpenRouter completion failed" in str(exc)
        assert "structured output failed" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("OpenRouterClient should fail fast on LangChain errors.")


def test_run_cli_fails_fast_when_llm_fails(tmp_path: Path, monkeypatch: Any) -> None:
    """Abort the run command and surface the failing LLM instead of faking output."""

    root_dir, _, _ = _build_demo_root(tmp_path)
    monkeypatch.setattr("news_system_demo.cli.ROOT_DIR", root_dir)
    monkeypatch.setattr("news_system_demo.cli.build_default_llm_client", lambda env_path: AlwaysFailLlmClient())
    runner = CliRunner()

    result = runner.invoke(app, ["--topic", "ai regulation europe", "--thread-id", "demo-thread"])

    assert result.exit_code == 1
    assert "news_system_demo FALLIDA" in result.stderr
    assert "forced llm failure" in result.stderr
    assert "run_log:" in result.stderr
