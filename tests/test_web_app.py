"""Юнит-тесты src/web/app.py — run_research замокан (офлайн, без реальных моделей).

Фоновый поток реального Job'а требует ожидания в тесте — polling с коротким
таймаутом, а не sleep вслепую.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from src.agent.research_runner import ResearchResult, SourceRef
from src.web import app as app_module


def _wait_until_done(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] != "running":
            return data
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def _reset_app_state(monkeypatch):
    monkeypatch.setattr(app_module, "_jobs", {})
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
            gaps=[],
            iterations=1,
            read_count=1,
            candidates_count=3,
        )

    monkeypatch.setattr(app_module, "run_research", fake_run_research)
    monkeypatch.setattr(app_module, "LanceDBStore", lambda: object())

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
    assert data["result"]["iterations"] == 1


def test_concurrent_job_is_rejected_with_409(monkeypatch):
    _reset_app_state(monkeypatch)

    started = __import__("threading").Event()
    release = __import__("threading").Event()

    def fake_run_research(question, store, on_progress=None):
        started.set()
        release.wait(timeout=5.0)
        return ResearchResult(
            answer="ok", sources=[], gaps=[], iterations=1, read_count=0, candidates_count=0
        )

    monkeypatch.setattr(app_module, "run_research", fake_run_research)
    monkeypatch.setattr(app_module, "LanceDBStore", lambda: object())

    client = TestClient(app_module.app)
    first = client.post("/api/jobs", json={"question": "first question"})
    assert first.status_code == 200
    started.wait(timeout=5.0)

    second = client.post("/api/jobs", json={"question": "second question"})
    assert second.status_code == 409

    release.set()
    _wait_until_done(client, first.json()["job_id"])


def test_job_reports_error_status_on_exception(monkeypatch):
    _reset_app_state(monkeypatch)

    def fake_run_research(question, store, on_progress=None):
        raise RuntimeError("something broke")

    monkeypatch.setattr(app_module, "run_research", fake_run_research)
    monkeypatch.setattr(app_module, "LanceDBStore", lambda: object())

    client = TestClient(app_module.app)
    job_id = client.post("/api/jobs", json={"question": "q"}).json()["job_id"]

    data = _wait_until_done(client, job_id)
    assert data["status"] == "error"
    assert "something broke" in data["error"]
