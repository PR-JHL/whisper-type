"""识别结果的后处理：去空格、修标点、去复读、过滤幻觉。"""

from __future__ import annotations

import re

CJK = r"\u4e00-\u9fff"
_CJK_RE = re.compile(f"[{CJK}]")
_HAN_SPACE = re.compile(f"(?<=[{CJK}])\\s+(?=[{CJK}])")
_HAN_ASCII = re.compile(f"(?<=[{CJK}])\\s+(?=[A-Za-z0-9])")
_ASCII_HAN = re.compile(f"(?<=[A-Za-z0-9])\\s+(?=[{CJK}])")

# Whisper 在静音/噪声上常见的"凭空字幕"
HALLUCINATIONS = {
    "谢谢观看",
    "谢谢观看。",
    "感谢观看",
    "感谢观看。",
    "请不吝点赞订阅转发打赏支持明镜与点点栏目",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
    "字幕由Amara.org社区提供",
    "字幕由 amara.org 社区提供",
    "字幕制作：",
    "下次再见",
    "下次再见。",
    "明镜与点点栏目",
    "请加标点",
    "请加标点。",
    "以下是普通话的句子",
    "以下是普通话的句子，请加标点。",
}


def prompt_echo(text: str, prompt: str) -> bool:
    """Whisper 在静音上会把 initial_prompt 原样吐出来（甚至复读），识别并丢掉。"""
    t = re.sub(r"[\s，。！？、,.!?]", "", text or "")
    p = re.sub(r"[\s，。！？、,.!?]", "", prompt or "")
    if not t:
        return True
    if not p:
        return False
    if t == p or (len(t) >= 4 and (t in p or p in t)):
        return True
    # "请加标点。请加标点。" 这种复读：字符集完全落在 prompt 里
    if len(t) >= 4 and set(t) <= set(p):
        return True
    return False


def _fix_ascii_punct_in_chinese(s: str) -> str:
    # 中文后面紧跟的半角标点换成全角（包括句末的情况）
    table = {",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；"}
    return re.sub(
        f"(?<=[{CJK}])([,.:;!?])", lambda m: table[m.group(1)], s
    )


def _collapse_repeats(s: str) -> str:
    # 同一个字连续超过 4 次 -> 留 2 个（口吃/幻觉）
    s = re.sub(r"(.)\1{3,}", r"\1\1", s)
    # 同一个 2~6 字短语连续重复 3 次以上 -> 留 2 次
    for n in range(2, 7):
        pat = re.compile(rf"((?:.{{{n}}})?)\1{{2,}}", re.S)
        s = pat.sub(lambda m: (m.group(1) or "") * 2 if m.group(1) else m.group(0), s)
    return s


def clean(text: str) -> str:
    s = (text or "").strip()
    s = s.replace("\u3000", " ")
    s = _HAN_SPACE.sub("", s)
    s = _fix_ascii_punct_in_chinese(s)
    # 中英文之间保留一个空格，读起来更舒服
    s = _HAN_ASCII.sub(" ", s)
    s = _ASCII_HAN.sub(" ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = _collapse_repeats(s)
    return s.strip()


def is_hallucination(text: str, audio_seconds: float) -> bool:
    s = re.sub(r"[\s，。！？、,.!?]", "", text or "")
    if not s:
        return True
    known = {re.sub(r"[\s，。！？、,.!?]", "", h) for h in HALLUCINATIONS}
    if audio_seconds <= 3.0 and s in known:
        return True
    # 短音频里凭空冒出的字幕/制作人信息
    if audio_seconds <= 3.0 and (s.startswith("字幕") or ("提供" in s and len(s) <= 12)):
        return True
    return False


def strip_punct(s: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:（）()“”\"'‘’—-]", "", s or "")