from src.agent import evaluate
from src.providers import rerank


def _chunks(*texts):
    return [{"text": t} for t in texts]


def test_no_citations_gives_zero_coverage(monkeypatch):
    def _fail_score_pairs(pairs):
        raise AssertionError("score_pairs must not be called with no citations")

    monkeypatch.setattr(rerank, "score_pairs", _fail_score_pairs)

    result = evaluate.evaluate("Кит — крупное морское млекопитающее.", _chunks("что-то"))
    assert result.citation_coverage == 0.0
    assert result.faithfulness == 0.0
    assert result.unsupported == []


def test_full_coverage_and_faithfulness_when_all_supported(monkeypatch):
    monkeypatch.setattr(rerank, "score_pairs", lambda pairs: [0.9 for _ in pairs])

    answer = "Кит — млекопитающее [1]. Живёт в океане [1]."
    result = evaluate.evaluate(answer, _chunks("Кит — крупное морское млекопитающее."))

    assert result.citation_coverage == 1.0
    assert result.faithfulness == 1.0
    assert result.unsupported == []


def test_catches_unsupported_claim(monkeypatch):
    # Первое утверждение поддержано источником, второе — выдумано (низкий score).
    monkeypatch.setattr(rerank, "score_pairs", lambda pairs: [0.9, 0.05])

    answer = "Кит — млекопитающее [1]. У китов три сердца и они дышат жабрами [1]."
    result = evaluate.evaluate(answer, _chunks("Кит — крупное морское млекопитающее."))

    assert result.citation_coverage == 1.0
    assert result.faithfulness == 0.5
    assert len(result.unsupported) == 1
    assert "жабрами" in result.unsupported[0].sentence


def test_partial_citation_coverage(monkeypatch):
    monkeypatch.setattr(rerank, "score_pairs", lambda pairs: [0.9])
    answer = "Кит — млекопитающее [1]. Это интересный факт."
    result = evaluate.evaluate(answer, _chunks("Кит — крупное морское млекопитающее."))
    assert result.citation_coverage == 0.5


def test_citation_pointing_outside_chunks_range_is_not_scored(monkeypatch):
    def _fail_score_pairs(pairs):
        raise AssertionError("score_pairs must not be called for out-of-range citation")

    monkeypatch.setattr(rerank, "score_pairs", _fail_score_pairs)

    answer = "Это утверждение ссылается на несуществующий источник [5]."
    result = evaluate.evaluate(answer, _chunks("текст"))
    assert result.checks[0].citations == [5]
    assert result.checks[0].faithful is None


def test_strips_models_own_sources_block(monkeypatch):
    monkeypatch.setattr(rerank, "score_pairs", lambda pairs: [0.9])

    answer = "Кит — млекопитающее [1].\n\nИсточники:\n[1] doc_about_whales"
    result = evaluate.evaluate(answer, _chunks("Кит — крупное морское млекопитающее."))

    assert len(result.checks) == 1
    assert "Источники" not in result.checks[0].sentence
