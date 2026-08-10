"""FastAPI веб-интерфейс поверх agent/research_runner.run_research()/run_followup().

Один research()-прогон может занимать минуты (реальные модели, внешние
источники) — выполняется в фоновом потоке, прогресс отдаётся клиенту через
polling (`GET /api/jobs/{id}`), без WebSocket/SSE — на масштабе одного
локального пользователя это ненужное усложнение.

Одновременно выполняется не больше одного research()-прогона (§1: нельзя
держать/грузить несколько тяжёлых моделей параллельно на 16ГБ) — новый запрос,
пока предыдущий не завершён, отклоняется 409, а не встаёт в очередь молча.

Follow-up-вопросы ("уточни", "раскрой подробнее тему N") продолжают тот же
диалог — `Session` хранит `ResearchState`/`QdrantStore` между ходами (job'ами)
одного разговора, см. `agent/research_runner.run_followup`. Первый вопрос
диалога создаёт сессию (`POST /api/jobs`), follow-up идёт в ту же сессию
(`POST /api/sessions/{session_id}/followup`) — оба возвращают job_id и
опрашиваются одинаково через `GET /api/jobs/{id}`.

Digest (`POST /api/digest` + `GET /api/digest/{id}`) — тот же
job+polling-паттерн, но своя, более простая пара `DigestJob`/`_digest_jobs`:
нет сессии/диалога/follow-up (`agent/digest.py` не возвращает `state` для
продолжения), значит незачем тащить сюда `Session`-обвязку `Job`. Общий с
research()-джобами только `_current_job_id`-слот (`_require_free_slot`) —
дайджест тоже дёргает резидентную LLM (для обзора тем), поэтому инвариант
"не больше одного тяжёлого прогона одновременно" (§1 CLAUDE.md) действует
на оба вида job'ов вместе, не по отдельности.
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

from .. import config
from ..agent.digest import DigestResult, run_digest
from ..agent.research_runner import ResearchResult, run_followup, run_research
from ..agent.state import ResearchState
from ..providers import tracing
from ..store.qdrant_store import QdrantStore

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# `cli.py serve` already calls this before `uvicorn.run` imports this module
# in-process — repeated here too so `uvicorn src.web.app:app` works standalone.
tracing.enable_if_configured()

app = FastAPI(title="local-research-agent")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@dataclass
class Job:
    id: str
    session_id: str
    question: str
    status: Literal["running", "done", "error"] = "running"
    progress: list[str] = field(default_factory=list)
    result: ResearchResult | None = None
    error: str | None = None


@dataclass
class Session:
    id: str
    store: QdrantStore
    state: ResearchState | None = None  # заполняется после первого завершённого хода


@dataclass
class DigestJob:
    id: str
    status: Literal["running", "done", "error"] = "running"
    progress: list[str] = field(default_factory=list)
    result: DigestResult | None = None
    error: str | None = None


_jobs: dict[str, Job] = {}
_sessions: dict[str, Session] = {}
_digest_jobs: dict[str, DigestJob] = {}
_jobs_lock = threading.Lock()
_current_job_id: str | None = None


class ResearchRequest(BaseModel):
    question: str


class FollowupRequest(BaseModel):
    question: str
    focus_candidate_id: str | None = None


class DigestRequest(BaseModel):
    days: int | None = None
    categories: list[str] | None = None
    limit: int | None = None
    summarize: bool | None = None
    query: str | None = None


def _job_to_dict(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "session_id": job.session_id,
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
            "candidates": [
                {
                    "id": c.id,
                    "title": c.title,
                    "source": c.source,
                    "url": c.url,
                    "citation_count": c.citation_count,
                    "triage_score": c.triage_score,
                    "read": c.read,
                }
                for c in job.result.candidates
            ],
            "gaps": job.result.gaps,
            "iterations": job.result.iterations,
            "read_count": job.result.read_count,
            "candidates_count": job.result.candidates_count,
        },
        "error": job.error,
    }


def _digest_job_to_dict(job: DigestJob) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": list(job.progress),
        "result": None
        if job.result is None
        else {
            "days": job.result.days,
            "categories": job.result.categories,
            "query": job.result.query,
            "summary": job.result.summary,
            "items": [
                {
                    "title": item.title,
                    "abstract": item.abstract,
                    "url": item.url,
                    "published_date": item.published_date,
                    "authors": item.meta.get("authors") or [],
                }
                for item in job.result.items
            ],
        },
        "error": job.error,
    }


def _run_digest_job(job: DigestJob, days: int | None, categories: list[str] | None,
                     limit: int | None, summarize: bool | None, query: str | None) -> None:
    global _current_job_id
    try:
        def on_progress(message: str) -> None:
            with _jobs_lock:
                job.progress.append(message)

        result = run_digest(days=days, categories=categories, limit=limit,
                             summarize=summarize, query=query, on_progress=on_progress)
        with _jobs_lock:
            job.result = result
            job.status = "done"
    except Exception as exc:
        with _jobs_lock:
            job.error = str(exc)
            job.status = "error"
    finally:
        with _jobs_lock:
            _current_job_id = None


def _run_job(job: Job, session: Session, run: Any) -> None:
    """`run` — `lambda on_progress: run_research(...)` либо
    `lambda on_progress: run_followup(...)`, оба возвращают `ResearchResult`
    с заполненным `.state`, который остаётся в `session` для следующего хода."""
    global _current_job_id
    try:
        def on_progress(message: str) -> None:
            with _jobs_lock:
                job.progress.append(message)

        result = run(on_progress)
        with _jobs_lock:
            job.result = result
            job.status = "done"
            session.state = result.state
    except Exception as exc:  # research() дошёл до пользователя как ошибка, не 500 без объяснения
        with _jobs_lock:
            job.error = str(exc)
            job.status = "error"
    finally:
        with _jobs_lock:
            _current_job_id = None


def _require_question(raw: str) -> str:
    question = raw.strip()
    if not question:
        raise HTTPException(400, "Пустой вопрос")
    return question


def _require_free_slot() -> None:
    """Вызывается уже под `_jobs_lock`. Держит инвариант "не больше одного
    тяжёлого прогона одновременно" (§1 CLAUDE.md — нельзя грузить/держать
    несколько тяжёлых моделей параллельно): общий для research()-джобов и
    digest-джобов, оба используют резидентную LLM."""
    if _current_job_id is not None:
        raise HTTPException(409, "Уже выполняется другой запрос — дождитесь его завершения")


def _claim_job_slot(session_id: str, question: str) -> Job:
    """Вызывается уже под `_jobs_lock`, после `_require_free_slot()` —
    заводит `Job` и сразу же занимает слот им."""
    global _current_job_id
    job = Job(id=str(uuid.uuid4()), session_id=session_id, question=question)
    _jobs[job.id] = job
    _current_job_id = job.id
    return job


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.post("/api/jobs")
def create_job(payload: ResearchRequest) -> dict[str, Any]:
    question = _require_question(payload.question)

    with _jobs_lock:
        _require_free_slot()
        session_id = str(uuid.uuid4())
        job = _claim_job_slot(session_id, question)
        # Отдельная таблица от `ask`/`index` — см. config.RESEARCH_INDEX_TABLE.
        # Создаётся только после успешного _claim_job_slot — если слот занят,
        # незачем заводить сессию, которой некому будет воспользоваться.
        session = Session(id=session_id, store=QdrantStore(collection_name=config.QDRANT_RESEARCH_COLLECTION))
        _sessions[session_id] = session

    run = lambda on_progress: run_research(job.question, session.store, on_progress=on_progress)
    threading.Thread(target=_run_job, args=(job, session, run), daemon=True).start()
    return {"job_id": job.id, "session_id": session_id}


@app.post("/api/sessions/{session_id}/followup")
def create_followup(session_id: str, payload: FollowupRequest) -> dict[str, Any]:
    question = _require_question(payload.question)

    with _jobs_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "Неизвестная сессия — сначала задайте исходный вопрос")
        if session.state is None:
            raise HTTPException(409, "Исходный запрос этой сессии ещё не завершён")
        _require_free_slot()
        job = _claim_job_slot(session_id, question)

    run = lambda on_progress: run_followup(
        job.question, session.state, session.store,
        on_progress=on_progress, focus_candidate_id=payload.focus_candidate_id,
    )
    threading.Thread(target=_run_job, args=(job, session, run), daemon=True).start()
    return {"job_id": job.id, "session_id": session_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Неизвестный job_id")
        return _job_to_dict(job)


@app.post("/api/digest")
def create_digest(payload: DigestRequest) -> dict[str, Any]:
    global _current_job_id
    with _jobs_lock:
        _require_free_slot()
        job = DigestJob(id=str(uuid.uuid4()))
        _digest_jobs[job.id] = job
        _current_job_id = job.id

    threading.Thread(
        target=_run_digest_job,
        args=(job, payload.days, payload.categories, payload.limit, payload.summarize, payload.query),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@app.get("/api/digest/{job_id}")
def get_digest(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _digest_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Неизвестный job_id")
        return _digest_job_to_dict(job)
