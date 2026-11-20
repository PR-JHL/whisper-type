import numpy as np

from whispertype import vad


def _tone(freq=440, seconds=1.0, sr=16000, amp=0.2):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_silence_has_no_speech():
    x = np.zeros(16000, dtype=np.float32)
    assert not vad.has_speech(x, 16000)


def test_tone_burst_detected():
    x = np.concatenate([np.zeros(8000, np.float32), _tone(seconds=0.5), np.zeros(8000, np.float32)])
    assert vad.has_speech(x, 16000)
    assert 0.2 < vad.speech_ratio(x, 16000) < 0.8


def test_trim_removes_leading_trailing_silence():
    x = np.concatenate([np.zeros(16000, np.float32), _tone(seconds=0.5), np.zeros(16000, np.float32)])
    y = vad.trim(x, 16000, margin_db=10, pad=0.05)
    assert len(y) < len(x)
    assert len(y) > 0.5 * 16000 * 0.9


def test_levels():
    x = _tone(seconds=1.0, amp=0.5)
    assert -10 < vad.rms_db(x) < 0
    assert vad.peak_db(x) > -8
    assert vad.noise_floor_db(np.zeros(1000, np.float32), 16000) < -80