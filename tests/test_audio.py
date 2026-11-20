import numpy as np

from whispertype import audio


def test_resample_length():
    x = np.zeros(48000, dtype=np.float32)
    y = audio.resample_to(x, 48000, 16000)
    assert len(y) == 16000


def test_resample_keeps_sine():
    sr, f = 48000, 440
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)
    y = audio.resample_to(x, sr, 16000)
    assert abs(len(y) - 16000) <= 2
    assert 0.3 < float(np.max(np.abs(y))) <= 0.55


def test_normalize_rms():
    x = np.full(16000, 0.01, dtype=np.float32)
    y = audio.normalize_rms(x, -20.0)
    assert abs(audio.vad.rms_db(y) + 20.0) < 1.0


def test_normalize_no_clip():
    x = np.ones(1000, dtype=np.float32)
    y = audio.normalize_rms(x, -3.0)
    assert float(np.max(np.abs(y))) <= 1.0


def test_wav_roundtrip(tmp_path):
    sr = 16000
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = tmp_path / "a.wav"
    audio.save_wav(p, x, sr)
    r = audio.load_wav(p)
    assert r.samplerate == sr
    assert len(r.samples) == len(x)
    assert np.allclose(r.samples, x, atol=1e-3)


def test_highpass_removes_dc():
    x = np.ones(16000, dtype=np.float32) * 0.5
    y = audio.highpass(x, 16000, 60.0)
    assert abs(float(np.mean(y))) < 1e-3


def test_prepare_returns_16k_and_normalized():
    from whispertype.config import Config

    sr = 48000
    t = np.arange(sr) / sr
    x = (0.01 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    y = audio.prepare(x, sr, Config())
    # 全段都是信号，裁剪后长度接近
    assert 15000 <= len(y) <= 16000
    assert audio.vad.rms_db(y) > -30