"""Юнит-тесты src/web/app.py — run_research замокан (офлайн, без реальных моделей).

Фоновый поток реального Job'а требует ожидания в тесте — polling с коротким
таймаутом, а не sleep вслепую.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from src.agent.digest import DigestResult
from src.agent.research_runner import CandidateSummary, ResearchResult, SourceRef
from src.agent.state import ResearchState
from src.sources.base import DiscoveredItem
from src.web import app as app_module


def _wait_until_done(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] != "running":
            return data
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def _wait_until_digest_done(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = client.get(f"/api/digest/{job_id}").json()
        if data["status"] != "running":
            return data
        time.sleep(0.02)
    raise AssertionError("digest job did not finish in time")


def _reset_app_state(monkeypatch):
    monkeypatch.setattr(app_module, "_jobs", {})
    monkeypatch.setattr(app_module, "_sessions", {})
    monkeypatch.setattr(app_module, "_digest_jobs", {})
    monkeypatch.setattr(app_module, "_current_job_id", None)


def test_index_serves_html(monkeypatch):
    _reset_app_state(monkeypatch)
    client = TestClient(app_module.app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "local-research-agent" in resp.text


def test_create_job_rejects_empty_question(monkeypatch):
    _reset_app_state(monkeypatch)
    client = TestClient(app_module.app)
    resp = client.post("/api/jobs", json={"question": "   "})
    assert resp.status_code == 400


def test_get_unknown_job_returns_404(monkeypatch):
    _reset_app_state(monkeypatch)
    client = TestClient(app_module.app)
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404


def test_full_job_lifecycle_reports_progress_and_result(monkeypatch):
    _reset_app_state(monkeypatch)

    def fake_run_research(question, store, on_progress=None):
        if on_progress:
            on_progress("Подвопросов: 1.")
            on_progress("Синтезируем ответ…")
        return ResearchResult(
            answer="Кит — млекопитающее [1].",
            sources=[SourceRef(title="whales.md", url="https://example.com/whales", citation_count=7)],
            candidates=[
                CandidateSummary(
                    id="c1", title="whales.md", source="web", url="https://example.com/whales",
                    citation_count=7, triage_score=0.87, read=True,
                ),
                CandidateSummary(id="c2", title="unrelated paper", source="arxiv", read=False),
            ],
            gaps=[],
            iterations=1,
            read_count=1,
            candidates_count=3,
        )

    monkeypatch.setattr(app_module, "run_research", fake_run_research)
    monkeypatch.setattr(app_module, "QdrantStore", lambda collection_name=None: object())

    client = TestClient(app_module.app)
    create_resp = client.post("/api/jobs", json={"question": "Кто такой кит?"})
    assert create_resp.status_code == 200
    job_id = create_resp.json()["job_id"]

    data = _wait_until_done(client, job_id)
    assert data["status"] == "done"
    assert "Подвопросов: 1." in data["progress"]
    assert data["result"]["answer"] == "Кит — млекопитающее [1]."
    assert data["result"]["sources"] == [
        {"title": "whales.md", "url": "https://example.com/whales", "citation_count": 7}
    ]
    assert data["result"]["candidates"] == [
        {
            "id": "c1", "title": "whales.md", "source": "web", "url": "https://example.com/whales",
            "citation_count": 7, "triage_score": 0.87, "read": True,
        },
        {
            "id": "c2", "title": "unrelated paper", "source": "arxiv", "url": "",
            "citation_count": None, "triage_score": None, "read": False,
        },
    ]
    assert data["result"]["iterations"] == 1


def test_concurrent_job_is_rejected_with_409(monkeypatch):
    _reset_app_state(monkeypatch)

    started = __import__("threading").Event()
    release = __import__("threading").Event()

    def fake_run_research(question, store, on_progress=None):
        started.set()
        release.wait(timeout=5.0)
        return ResearchResult(
            answer="ok", sources=[], candidates=[], gaps=[], iterations=1, read_count=0,
            candidates_count=0,
        )

    monkeypatch.setattr(app_module, "run_research", fake_run_research)
    monkeypatch.setattr(app_module, "QdrantStore", lambda collection_name=None: object())

    client = TestClient(app_module.app)
    first = client.post("/api/jobs", json={"question": "first question"})
    assert first.status_code == 200
    started.wait(timeout=5.0)

    second = client.post("/api/jobs", json={"question": "second question"})
    assert second.status_code == 409

    release.set()
    _wait_until_done(client, first.json()["job_id"])


def test_followup_rejects_unknown_session(monkeypatch):
    _reset_app_state(monkeypatch)
    client = TestClient(app_module.app)
    resp = client.post("/api/sessions/does-not-exist/followup", json={"question": "а что насчёт X?"})
    assert resp.status_code == 404


def test_followup_continues_session_after_initial_job(monkeypatch):
    _reset_app_state(monkeypatch)

    def fake_run_research(question, store, on_progress=None):
        return ResearchResult(
            answer="Кит — млекопитающее [1].",
            sources=[], candidates=[], gaps=[], iterations=1, read_count=1,
            candidates_count=1, state=ResearchState(question=question),
        )

    followup_calls = []

    def fake_run_followup(question, state, store, on_progress=None, focus_candidate_id=None):
        followup_calls.append((question, state, focus_candidate_id))
        return ResearchResult(
            answer="А самки крупнее самцов [1].",
            sources=[], candidates=[], gaps=[], iterations=1, read_count=1,
            candidates_count=1, state=state,
        )

    monkeypatch.setattr(app_module, "run_research", fake_run_research)
    monkeypatch.setattr(app_module, "run_followup", fake_run_followup)
    monkeypatch.setattr(app_module, "QdrantStore", lambda collection_name=None: object())

    client = TestClient(app_module.app)
    create_resp = client.post("/api/jobs", json={"question": "Кто такой кит?"})
    session_id = create_resp.json()["session_id"]
    _wait_until_done(client, create_resp.json()["job_id"])

    followup_resp = client.post(
        f"/api/sessions/{session_id}/followup",
        json={"question": "а самки крупнее?", "focus_candidate_id": "c1"},
    )
    assert followup_resp.status_code == 200
    data = _wait_until_done(client, followup_resp.json()["job_id"])

    assert data["status"] == "done"
    assert data["result"]["answer"] == "А самки крупнее самцов [1]."
    assert len(followup_calls) == 1
    question, state, focus_id = followup_calls[0]
    assert question == "а самки крупнее?"
    assert focus_id == "c1"
    assert state.question == "Кто такой кит?"  # тот же ResearchState, не пересоздан


def test_job_reports_error_status_on_exception(monkeypatch):
    _reset_app_state(monkeypatch)

    def fake_run_research(question, store, on_progress=None):
        raise RuntimeError("something broke")

    monkeypatch.setattr(app_module, "run_research", fake_run_research)
    monkeypatch.setattr(app_module, "QdrantStore", lambda collection_name=None: object())

    client = TestClient(app_module.app)
    job_id = client.post("/api/jobs", json={"question": "q"}).json()["job_id"]

    data = _wait_until_done(client, job_id)
    assert data["status"] == "error"
    assert "something broke" in data["error"]


def test_get_unknown_digest_returns_404(monkeypatch):
    _reset_app_state(monkeypatch)
    client = TestClient(app_module.app)
    resp = client.get("/api/digest/does-not-exist")
    assert resp.status_code == 404


def test_full_digest_lifecycle_reports_progress_and_result(monkeypatch):
    _reset_app_state(monkeypatch)

    def fake_run_digest(days=None, categories=None, limit=None, summarize=None, query=None, on_progress=None):
        if on_progress:
            on_progress("Ищем статьи…")
        item = DiscoveredItem(
            id="arxiv:1", source="arxiv", title="Paper", abstract="Abstract",
            url="https://arxiv.org/abs/1", published_date="2026-08-07T00:00:00Z",
            meta={"authors": ["A. Uthor"]},
        )
        return DigestResult(items=[item], days=7, categories=["cs.AI"], query=query, summary="Overview.")

    monkeypatch.setattr(app_module, "run_digest", fake_run_digest)

    client = TestClient(app_module.app)
    create_resp = client.post("/api/digest", json={"days": 7, "query": "diffusion"})
    assert create_resp.status_code == 200
    job_id = create_resp.json()["job_id"]

    data = _wait_until_digest_done(client, job_id)
    assert data["status"] == "done"
    assert "Ищем статьи…" in data["progress"]
    assert data["result"]["summary"] == "Overview."
    assert data["result"]["days"] == 7
    assert data["result"]["categories"] == ["cs.AI"]
    assert data["result"]["query"] == "diffusion"
    assert data["result"]["items"] == [
        {
            "title": "Paper", "abstract": "Abstract", "url": "https://arxiv.org/abs/1",
            "published_date": "2026-08-07T00:00:00Z", "authors": ["A. Uthor"],
        }
    ]


def test_digest_job_reports_error_status_on_exception(monkeypatch):
    _reset_app_state(monkeypatch)

    def fake_run_digest(**kwargs):
        raise RuntimeError("arxiv is down")

    monkeypatch.setattr(app_module, "run_digest", fake_run_digest)

    client = TestClient(app_module.app)
    job_id = client.post("/api/digest", json={}).json()["job_id"]

    data = _wait_until_digest_done(client, job_id)
    assert data["status"] == "error"
    assert "arxiv is down" in data["error"]


def test_digest_shares_the_same_concurrency_slot_as_research(monkeypatch):
    """Дайджест тоже дёргает резидентную LLM (обзор тем) — инвариант
    "не больше одного тяжёлого прогона одновременно" общий с research()."""
    _reset_app_state(monkeypatch)

    started = __import__("threading").Event()
    release = __import__("threading").Event()

    def fake_run_research(question, store, on_progress=None):
        started.set()
        release.wait(timeout=5.0)
        return ResearchResult(
            answer="ok", sources=[], candidates=[], gaps=[], iterations=1, read_count=0,
            candidates_count=0,
        )

    monkeypatch.setattr(app_module, "run_research", fake_run_research)
    monkeypatch.setattr(app_module, "QdrantStore", lambda collection_name=None: object())

    client = TestClient(app_module.app)
    first = client.post("/api/jobs", json={"question": "first question"})
    assert first.status_code == 200
    started.wait(timeout=5.0)

    digest_resp = client.post("/api/digest", json={})
    assert digest_resp.status_code == 409

    release.set()
    _wait_until_done(client, first.json()["job_id"])
