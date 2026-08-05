from src.providers import rerank


def test_rerank_sorts_by_score_descending(monkeypatch):
    candidates = [
        {"text": "low relevance", "source_id": "a"},
        {"text": "high relevance", "source_id": "b"},
        {"text": "medium relevance", "source_id": "c"},
    ]
    score_by_text = {"low relevance": 0.1, "high relevance": 0.9, "medium relevance": 0.5}
    monkeypatch.setattr(rerank, "_load", lambda: (object(), object()))
    monkeypatch.setattr(rerank, "_release", lambda model, tokenizer: None)
    monkeypatch.setattr(
        rerank,
        "_score_batch",
        lambda model, tokenizer, query, texts: [score_by_text[t] for t in texts],
    )

    result = rerank.rerank("query", candidates, top_n=3)
    assert [c["source_id"] for c in result] == ["b", "c", "a"]


def test_rerank_truncates_to_top_n(monkeypatch):
    candidates = [
        {"text": "a", "source_id": "a"},
        {"text": "b", "source_id": "b"},
    ]
    monkeypatch.setattr(rerank, "_load", lambda: (object(), object()))
    monkeypatch.setattr(rerank, "_release", lambda model, tokenizer: None)
    monkeypatch.setattr(
        rerank, "_score_batch", lambda model, tokenizer, query, texts: [0.2, 0.8]
    )

    result = rerank.rerank("query", candidates, top_n=1)
    assert [c["source_id"] for c in result] == ["b"]


def test_rerank_releases_model_even_on_scoring_error(monkeypatch):
    released = []
    monkeypatch.setattr(rerank, "_load", lambda: ("model", "tokenizer"))
    monkeypatch.setattr(rerank, "_release", lambda model, tokenizer: released.append((model, tokenizer)))

    def _raise(model, tokenizer, query, texts):
        raise RuntimeError("boom")

    monkeypatch.setattr(rerank, "_score_batch", _raise)

    try:
        rerank.rerank("query", [{"text": "x", "source_id": "x"}], top_n=1)
    except RuntimeError:
        pass
    assert released == [("model", "tokenizer")]


def test_rerank_empty_candidates_skips_model_load(monkeypatch):
    def _fail_load():
        raise AssertionError("_load must not be called for empty candidates")

    monkeypatch.setattr(rerank, "_load", _fail_load)
    assert rerank.rerank("query", [], top_n=5) == []


def test_score_pairs_scores_each_pair_with_its_own_query(monkeypatch):
    monkeypatch.setattr(rerank, "_load", lambda: (object(), object()))
    monkeypatch.setattr(rerank, "_release", lambda model, tokenizer: None)
    monkeypatch.setattr(
        rerank, "_score_one", lambda model, tokenizer, query, text: len(query) + len(text)
    )

    scores = rerank.score_pairs([("q1", "aa"), ("q22", "b")])
    assert scores == [len("q1") + len("aa"), len("q22") + len("b")]


def test_score_pairs_empty_skips_model_load(monkeypatch):
    def _fail_load():
        raise AssertionError("_load must not be called for empty pairs")

    monkeypatch.setattr(rerank, "_load", _fail_load)
    assert rerank.score_pairs([]) == []


def test_score_pairs_releases_model_even_on_scoring_error(monkeypatch):
    released = []
    monkeypatch.setattr(rerank, "_load", lambda: ("model", "tokenizer"))
    monkeypatch.setattr(rerank, "_release", lambda model, tokenizer: released.append((model, tokenizer)))
    monkeypatch.setattr(
        rerank, "_score_one", lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    try:
        rerank.score_pairs([("q", "t")])
    except RuntimeError:
        pass
    assert released == [("model", "tokenizer")]
