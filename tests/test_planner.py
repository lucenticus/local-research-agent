from src.agent.planner import plan


def test_empty_question_returns_no_subquestions():
    assert plan("") == []
    assert plan("   ") == []


def test_single_question_stays_one_subquestion():
    result = plan("Что такое RAG?")
    assert [sq.text for sq in result] == ["Что такое RAG?"]


def test_compound_question_splits_on_question_marks():
    result = plan("Что такое RAG? Зачем нужен hybrid retrieval?")
    assert [sq.text for sq in result] == ["Что такое RAG?", "Зачем нужен hybrid retrieval?"]


def test_splits_on_a_takzhe_separator():
    result = plan("Что такое BM25, а также как работает RRF?")
    # каждый подвопрос гарантированно заканчивается "?", даже если в исходном
    # тексте разделитель его "съел"
    assert [sq.text for sq in result] == ["Что такое BM25?", "как работает RRF?"]
