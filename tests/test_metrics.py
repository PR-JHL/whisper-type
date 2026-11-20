from whispertype.metrics import cer, edit_distance, report


def test_edit_distance():
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("", "abc") == 3


def test_cer_perfect():
    assert cer("今天天气不错", "今天天气不错。") == 0.0


def test_cer_one_error():
    r = report("今天天气不错", "今天天汽不错")
    assert r["edits"] == 1
    assert abs(r["cer"] - 1 / 6) < 1e-9


def test_cer_ignores_punct_and_space():
    assert cer("你好，世界", "你 好 世 界") == 0.0