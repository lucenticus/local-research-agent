"""FastAPI веб-интерфейс поверх agent/research_runner.run_research().

Один research()-прогон может занимать минуты (реальные модели, внешние
источники) — выполняется в фоновом потоке, прогресс отдаётся клиенту через
polling (`GET /api/jobs/{id}`), без WebSocket/SSE — на масштабе одного
локального пользователя это ненужное усложнение.

Одновременно выполняется не больше одного research()-прогона (§1: нельзя
держать/грузить несколько тяжёлых моделей параллельно на 16ГБ) — новый запрос,
пока предыдущий не завершён, отклоняется 409, а не встаёт в очередь молча.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..agent.research_runner import ResearchResult, run_research
from ..store.lancedb_store import LanceDBStore

_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="local-research-agent")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@dataclass
class Job:
    id: str
    question: str
    status: Literal["running", "done", "error"] = "running"
    progress: list[str] = field(default_factory=list)
    result: ResearchResult | None = None
    error: str | None = None


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
_current_job_id: str | None = None


class ResearchRequest(BaseModel):
    question: str


def _job_to_dict(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "question": job.question,
        "status": job.status,
        "progress": list(job.progress),
        "result": None
        if job.result is None
        else {
            "answer": job.result.answer,
            "sources": [
                {"title": s.title, "url": s.url, "citation_count": s.citation_count}
                for s in job.result.sources
            ],
            "gaps": job.result.gaps,
            "iterations": job.result.iterations,
            "read_count": job.result.read_count,
            "candidates_count": job.result.candidates_count,
        },
        "error": job.error,
    }


def _run_job(job: Job) -> None:
    global _current_job_id
    try:
        def on_progress(message: str) -> None:
            with _jobs_lock:
                job.progress.append(message)

        store = LanceDBStore()
        result = run_research(job.question, store, on_progress=on_progress)
        with _jobs_lock:
            job.result = result
            job.status = "done"
    except Exception as exc:  # research() дошёл до пользователя как ошибка, не 500 без объяснения
        with _jobs_lock:
            job.error = str(exc)
            job.status = "error"
    finally:
        with _jobs_lock:
            _current_job_id = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.post("/api/jobs")
def create_job(payload: ResearchRequest) -> dict[str, Any]:
    global _current_job_id
    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "Пустой вопрос")

    with _jobs_lock:
        if _current_job_id is not None:
            raise HTTPException(
                409, "Уже выполняется другой research-запрос — дождитесь его завершения"
            )
        job = Job(id=str(uuid.uuid4()), question=question)
        _jobs[job.id] = job
        _current_job_id = job.id

    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Неизвестный job_id")
        return _job_to_dict(job)
