"""录音、重采样、归一化、存 WAV。"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sps

from . import vad
from .config import Config

TARGET_SR = 16000


def _sd():
    import sounddevice as sd  # 延迟导入：没装声卡库时其它模块仍可导入

    return sd


def list_input_devices() -> list[dict]:
    sd = _sd()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append(
                {
                    "index": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "samplerate": int(d["default_samplerate"]),
                    "hostapi": sd.query_hostapis(d["hostapi"])["name"],
                }
            )
    return out


def default_input_index() -> int | None:
    sd = _sd()
    try:
        d = sd.query_devices(kind="input")
        for i, dev in enumerate(sd.query_devices()):
            if dev["name"] == d["name"] and dev["hostapi"] == d["hostapi"]:
                return i
    except Exception:
        pass
    return None


def resolve_device(spec: int | str | None) -> int | None:
    """int 直接用；字符串按名称模糊匹配；None 用系统默认。"""
    if spec is None or spec == "":
        return None
    if isinstance(spec, int):
        return spec
    s = str(spec).strip()
    if s.isdigit():
        return int(s)
    devices = list_input_devices()
    matches = [d for d in devices if s.lower() in d["name"].lower()]
    if not matches:
        raise ValueError(f"找不到包含 {s!r} 的输入设备；用 `whispertype devices` 查看列表")
    names = {d["name"] for d in matches}
    if len(matches) > 1 and len(names) > 1:
        listing = "、".join(f"[{d['index']}] {d['name']}" for d in matches)
        raise ValueError(f"{s!r} 匹配到多个设备：{listing}，请改用序号")
    # 同一个麦克风会被 MME/DirectSound/WASAPI/WDM-KS 各列一遍，优先 WASAPI
    pref = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2, "Windows WDM-KS": 3}
    matches.sort(key=lambda d: pref.get(d["hostapi"], 9))
    return matches[0]["index"]


@dataclass
class Recording:
    samples: np.ndarray  # float32 mono
    samplerate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / max(1, self.samplerate)


class Recorder:
    """按住说话用的录音器：start() 开始，stop() 返回整段。"""

    def __init__(self, device: int | None = None, samplerate: int | None = None) -> None:
        self.device = device
        self.samplerate = samplerate
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = None

    def _pick_samplerate(self) -> int:
        sd = _sd()
        info = sd.query_devices(self.device, "input")
        default_sr = int(info["default_samplerate"])
        if not self.samplerate:
            return default_sr
        try:
            sd.check_input_settings(
                device=self.device, channels=1, samplerate=self.samplerate, dtype="float32"
            )
            return self.samplerate
        except Exception:
            return default_sr

    def start(self) -> None:
        import threading

        if self._stream is not None:
            return
        sd = _sd()
        self._lock = threading.Lock()
        self._samplerate = self._pick_samplerate()
        self._chunks = []

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            with self._lock:
                self._chunks.append(indata[:, 0].copy())

        self._stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self._samplerate,
            dtype="float32",
            blocksize=1024,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> Recording:
        if self._stream is None:
            return Recording(np.zeros(0, dtype=np.float32), self.samplerate or TARGET_SR)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            chunks = self._chunks
            self._chunks = []
        x = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return Recording(x.astype(np.float32), int(self._samplerate))

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def seconds(self) -> float:
        if self._lock is None:
            return 0.0
        with self._lock:
            n = sum(len(c) for c in self._chunks)
        return n / max(1, getattr(self, "_samplerate", TARGET_SR))


def resample_to(x: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32, copy=False)
    g = np.gcd(int(sr_in), int(sr_out))
    y = sps.resample_poly(x, sr_out // g, sr_in // g)
    return np.asarray(y, dtype=np.float32)


def highpass(x: np.ndarray, sr: int, hz: float = 60.0, order: int = 2) -> np.ndarray:
    if len(x) < 32 or hz <= 0:
        return x
    sos = sps.butter(order, hz, btype="highpass", fs=sr, output="sos")
    return sps.sosfilt(sos, x).astype(np.float32)


def normalize_rms(x: np.ndarray, target_dbfs: float = -20.0, max_gain_db: float = 36.0) -> np.ndarray:
    if len(x) == 0:
        return x
    cur = vad.rms_db(x)
    gain_db = target_dbfs - cur
    gain_db = min(gain_db, max_gain_db)
    gain = 10 ** (gain_db / 20.0)
    y = x * gain
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:  # 防削波
        y = y * (0.98 / peak)
    return y.astype(np.float32)


def prepare(x: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    """原始录音 -> 16k 单声道、去首尾静音、归一化。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) == 0:
        return x
    x = resample_to(x, sr, TARGET_SR)
    x = highpass(x, TARGET_SR, 60.0)
    trimmed = vad.trim(x, TARGET_SR, margin_db=cfg.vad_margin_db, pad=0.12)
    # 如果整段都是"说话"（没有静音作参考），trim 可能把整段判没，此时保留原段
    if len(trimmed) >= int(0.1 * TARGET_SR):
        x = trimmed
    x = normalize_rms(x, cfg.target_dbfs)
    return x


def save_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    x = np.asarray(x, dtype=np.float32)
    pcm = np.clip(x, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm16.tobytes())


def load_wav(path: str | Path) -> Recording:
    with wave.open(str(path), "rb") as w:
        return _decode_wave(w)


def decode_wav_bytes(data: bytes) -> Recording:
    """从内存里的 WAV 字节解码（DSH 传输的就是这种 16k 单声道 PCM16）。"""
    import io

    with wave.open(io.BytesIO(data), "rb") as w:
        return _decode_wave(w)


def _decode_wave(w) -> Recording:  # noqa: ANN001
    sr = w.getframerate()
    n = w.getnframes()
    ch = w.getnchannels()
    width = w.getsampwidth()
    raw = w.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"不支持的位深: {width*8} bit")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return Recording(data.astype(np.float32), sr)