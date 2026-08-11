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
from ..agent.digest import DigestResult, ItemAnalysis, analyze_item, run_digest
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
    item_analyses: dict[str, "ItemAnalysisJob"] = field(default_factory=dict)  # ключ — item.id


@dataclass
class ItemAnalysisJob:
    """Анализ одной статьи дайджеста по клику ("Детальный анализ" в UI),
    отдельно от `DigestJob.result.analyses` (bulk-режим `deep=True`, CLI) —
    веб-UI больше не гонит анализ по всем статьям разом, только по запросу
    пользователя на конкретную карточку."""

    status: Literal["running", "done", "error"] = "running"
    progress: list[str] = field(default_factory=list)
    result: ItemAnalysis | None = None
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
    deep: bool = False


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


def _analysis_to_dict(analysis: Any) -> dict[str, Any] | None:
    if analysis is None:
        return None
    details = analysis.details
    insights = analysis.insights
    return {
        "summary_ru": analysis.summary_ru,
        "details": None
        if details is None
        else {
            "citation_count": details.citation_count,
            "venue": details.venue,
            "authors": [
                {"name": a.name, "institution": a.institution, "h_index": a.h_index} for a in details.authors
            ],
        },
        "insights": None
        if insights is None
        else {
            "findings_ru": insights.findings_ru,
            "code_links": [
                {"url": link.url, "kind": link.kind, "stars": link.stars}
                for link in insights.code_links
            ],
            "sections_used": list(insights.sections_used),
            "authors": [
                {"name": a.name, "affiliation": a.affiliation} for a in insights.authors
            ],
        },
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
                    "id": item.id,
                    "title": item.title,
                    "abstract": item.abstract,
                    "url": item.url,
                    "published_date": item.published_date,
                    "authors": item.meta.get("authors") or [],
                    "analysis": _analysis_to_dict(job.result.analyses.get(item.id)),
                }
                for item in job.result.items
            ],
        },
        "error": job.error,
    }


def _item_analysis_job_to_dict(job: ItemAnalysisJob) -> dict[str, Any]:
    return {
        "status": job.status,
        "progress": list(job.progress),
        "result": _analysis_to_dict(job.result),
        "error": job.error,
    }


def _find_digest_item(job: DigestJob, item_id: str) -> Any:
    if job.result is None:
        raise HTTPException(409, "The digest is not ready yet")
    for item in job.result.items:
        if item.id == item_id:
            return item
    raise HTTPException(404, "Unknown paper in this digest")


def _run_item_analysis_job(digest_job: DigestJob, item_id: str, item: Any) -> None:
    global _current_job_id
    analysis_job = digest_job.item_analyses[item_id]
    try:
        def on_progress(message: str) -> None:
            with _jobs_lock:
                analysis_job.progress.append(message)

        result = analyze_item(item, on_progress=on_progress)
        with _jobs_lock:
            analysis_job.result = result
            analysis_job.status = "done"
    except Exception as exc:
        with _jobs_lock:
            analysis_job.error = str(exc)
            analysis_job.status = "error"
    finally:
        with _jobs_lock:
            _current_job_id = None


def _run_digest_job(job: DigestJob, days: int | None, categories: list[str] | None,
                     limit: int | None, summarize: bool | None, query: str | None, deep: bool) -> None:
    global _current_job_id
    try:
        def on_progress(message: str) -> None:
            with _jobs_lock:
                job.progress.append(message)

        result = run_digest(days=days, categories=categories, limit=limit,
                             summarize=summarize, query=query, deep=deep, on_progress=on_progress)
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
        raise HTTPException(400, "Empty question")
    return question


def _require_free_slot() -> None:
    """Вызывается уже под `_jobs_lock`. Держит инвариант "не больше одного
    тяжёлого прогона одновременно" (§1 CLAUDE.md — нельзя грузить/держать
    несколько тяжёлых моделей параллельно): общий для research()-джобов и
    digest-джобов, оба используют резидентную LLM."""
    if _current_job_id is not None:
        raise HTTPException(409, "Another job is already running — wait for it to finish")


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
            raise HTTPException(404, "Unknown session — ask the initial question first")
        if session.state is None:
            raise HTTPException(409, "This session's initial job has not finished yet")
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
            raise HTTPException(404, "Unknown job_id")
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
        args=(job, payload.days, payload.categories, payload.limit, payload.summarize, payload.query, payload.deep),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@app.get("/api/digest/{job_id}")
def get_digest(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _digest_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        return _digest_job_to_dict(job)


@app.post("/api/digest/{job_id}/items/{item_id}/analyze")
def create_item_analysis(job_id: str, item_id: str) -> dict[str, Any]:
    global _current_job_id
    with _jobs_lock:
        digest_job = _digest_jobs.get(job_id)
        if digest_job is None:
            raise HTTPException(404, "Unknown job_id")
        item = _find_digest_item(digest_job, item_id)
        existing = digest_job.item_analyses.get(item_id)
        if existing is not None and existing.status == "running":
            raise HTTPException(409, "Analysis of this paper is already running")
        _require_free_slot()
        analysis_job = ItemAnalysisJob()
        digest_job.item_analyses[item_id] = analysis_job
        _current_job_id = f"{job_id}:{item_id}"

    threading.Thread(
        target=_run_item_analysis_job, args=(digest_job, item_id, item), daemon=True
    ).start()
    return {"status": "running"}


@app.get("/api/digest/{job_id}/items/{item_id}/analyze")
def get_item_analysis(job_id: str, item_id: str) -> dict[str, Any]:
    with _jobs_lock:
        digest_job = _digest_jobs.get(job_id)
        if digest_job is None:
            raise HTTPException(404, "Unknown job_id")
        analysis_job = digest_job.item_analyses.get(item_id)
        if analysis_job is None:
            raise HTTPException(404, "Analysis of this paper has not been started")
        return _item_analysis_job_to_dict(analysis_job)
