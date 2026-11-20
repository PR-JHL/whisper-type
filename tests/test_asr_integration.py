"""真实模型集成测试：默认跳过，设置 RUN_ASR=1 才跑（需要已下载模型）。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from whispertype.metrics import cer

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_ASR"), reason="设置 RUN_ASR=1 才跑（需要 faster-whisper 和模型）"
)

DATA = Path(__file__).parent / "data"
REF = "今天天气不错，我们下午三点开会，请把电脑带上。"


def test_transcribe_tts_sample():
    from whispertype.asr import ASR
    from whispertype.config import Config

    cfg = Config().merged(model="tiny", compute_device="cpu")
    res = ASR(cfg).transcribe_file(DATA / "tts_zh.wav")
    assert res.text.strip()
    assert cer(REF, res.text) < 0.2, res.text
    assert res.rtf < 1.0