from whispertype.textproc import clean, is_hallucination, prompt_echo, strip_punct


def test_remove_spaces_between_chinese():
    assert clean("你 好 世 界") == "你好世界"


def test_keep_space_between_chinese_and_english():
    assert clean("打开 Python 文件") == "打开 Python 文件"


def test_ascii_punct_to_chinese_in_context():
    assert clean("今天天气不错,我们去公园") == "今天天气不错，我们去公园"
    assert clean("你吃了吗?") == "你吃了吗？"


def test_collapse_repeated_chars():
    assert clean("谢谢谢谢谢谢谢谢") == "谢谢"
    assert "好好好" not in clean("好好好好好")


def test_strip_leading_trailing():
    assert clean("  你好  ") == "你好"


def test_hallucination_filter():
    assert is_hallucination("谢谢观看", 2.0)
    assert is_hallucination("", 2.0)
    assert not is_hallucination("今天开会", 2.0)
    assert not is_hallucination("谢谢观看", 10.0)  # 长音频里出现就可能是真话


def test_strip_punct():
    assert strip_punct("你好，世界！") == "你好世界"


def test_prompt_echo_filter():
    prompt = "以下是普通话的句子，请加标点。"
    assert prompt_echo("请加标点。请加标点。", prompt)
    assert prompt_echo("以下是普通话的句子", prompt)
    assert not prompt_echo("今天下午三点开会", prompt)


def test_short_audio_subtitle_hallucination():
    assert is_hallucination("字幕by索兰娅", 1.5)
    assert not is_hallucination("字幕by索兰娅", 8.0)