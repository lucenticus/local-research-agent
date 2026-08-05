from src.agent.state import Budget, Candidate, ResearchState, SubQuestion, SubQuestionStatus


def _candidate(id_: str) -> Candidate:
    return Candidate(id=id_, source="test", title=id_, abstract="abstract")


def test_add_candidates_dedups_by_id():
    state = ResearchState(question="q")
    added_first = state.add_candidates([_candidate("a"), _candidate("b")])
    added_second = state.add_candidates([_candidate("b"), _candidate("c")])

    assert [c.id for c in added_first] == ["a", "b"]
    assert [c.id for c in added_second] == ["c"]
    assert [c.id for c in state.candidates] == ["a", "b", "c"]


def test_mark_read_and_is_read():
    state = ResearchState(question="q")
    assert not state.is_read("a")
    state.mark_read("a")
    assert state.is_read("a")
    state.mark_read("a")  # идемпотентно
    assert state.read_ids == {"a"}


def test_budget_exhausted_by_iterations():
    state = ResearchState(question="q", budget=Budget(max_iterations=2, max_deep_reads=100, max_seconds=None))
    assert not state.budget_exhausted()
    state.iterations = 2
    assert state.budget_exhausted()


def test_budget_exhausted_by_deep_reads():
    state = ResearchState(question="q", budget=Budget(max_iterations=100, max_deep_reads=2, max_seconds=None))
    state.mark_read("a")
    assert not state.budget_exhausted()
    state.mark_read("b")
    assert state.budget_exhausted()


def test_budget_exhausted_by_time(monkeypatch):
    import src.agent.state as state_module

    state = ResearchState(question="q", budget=Budget(max_iterations=100, max_deep_reads=100, max_seconds=3.0))
    # default_factory=time.monotonic захватывает функцию на момент импорта
    # модуля (до monkeypatch) - выставляем _started_at напрямую, чтобы тест
    # управлял обеими точками отсчёта.
    state._started_at = 100.0
    monkeypatch.setattr(state_module.time, "monotonic", lambda: 105.0)

    assert state.budget_exhausted()


def test_cover_updates_status_and_open_sub_questions():
    state = ResearchState(question="q")
    state.sub_questions = [SubQuestion(text="a"), SubQuestion(text="b")]

    assert [sq.text for sq in state.open_sub_questions()] == ["a", "b"]
    state.cover("a")
    assert state.sub_questions[0].status == SubQuestionStatus.COVERED
    assert [sq.text for sq in state.open_sub_questions()] == ["b"]


def test_add_gap_deduplicates():
    state = ResearchState(question="q")
    state.add_gap("x")
    state.add_gap("x")
    assert state.gaps == ["x"]
