"""极简能量 VAD：够用、零依赖，专门处理"近距离气声"这种信噪比还行的场景。"""

from __future__ import annotations

import numpy as np


def frame_rms(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000))
    if len(x) < n:
        return np.array([float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))])
    usable = len(x) // n * n
    frames = x[:usable].astype(np.float64).reshape(-1, n)
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)


def to_db(v: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(v, 1e-9))


def noise_floor_db(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> float:
    rms = frame_rms(x, sr, frame_ms)
    if rms.size == 0:
        return -90.0
    return float(np.percentile(to_db(rms), 20))


def peak_db(x: np.ndarray) -> float:
    return float(to_db(float(np.max(np.abs(x))) if len(x) else 0.0))


def rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -90.0
    return float(to_db(float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))))


def trim(
    x: np.ndarray,
    sr: int,
    margin_db: float = 10.0,
    pad: float = 0.12,
    absolute_floor_db: float = -55.0,
) -> np.ndarray:
    """去掉首尾静音，保留 pad 秒的余量。"""
    if len(x) == 0:
        return x
    rms = frame_rms(x, sr)
    db = to_db(rms)
    floor = float(np.percentile(db, 20))
    thr = max(floor + margin_db, absolute_floor_db)
    idx = np.flatnonzero(db > thr)
    if idx.size == 0:
        return x[:0]
    frame_n = max(1, int(sr * 0.02))
    start = max(0, idx[0] * frame_n - int(pad * sr))
    end = min(len(x), (idx[-1] + 1) * frame_n + int(pad * sr))
    return x[start:end]


def has_speech(
    x: np.ndarray, sr: int, margin_db: float = 10.0, absolute_floor_db: float = -55.0
) -> bool:
    if len(x) == 0:
        return False
    floor = noise_floor_db(x, sr)
    peak = float(np.max(to_db(frame_rms(x, sr))))
    return peak > max(floor + margin_db, absolute_floor_db)


def speech_ratio(x: np.ndarray, sr: int, margin_db: float = 10.0) -> float:
    """说话帧占比，用来判断"是不是真的在说"。"""
    if len(x) == 0:
        return 0.0
    db = to_db(frame_rms(x, sr))
    floor = float(np.percentile(db, 20))
    return float(np.mean(db > floor + margin_db))