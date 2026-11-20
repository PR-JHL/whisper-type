"""SenseVoiceSmall（sherpa-onnx）识别引擎。

为什么加它：真实气声上 faster-whisper small 的 CER 是 70.8%，SenseVoiceSmall 是 50.1%，
而 SenseVoice 在 CPU 上还更快（ONNX int8）。所以默认引擎换成 SenseVoice，whisper 保留可选。

模型：csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17（约 230MB）
第一次用会下载到 ~/.whispertype/models/sensevoice/。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .asr import ASRResult
from .audio import TARGET_SR, resample_to
from .config import MODELS_DIR, Config

SENSEVOICE_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


def model_dir() -> Path:
    return MODELS_DIR / "sensevoice"


def ensure_model(cfg: Config | None = None) -> Path:
    """模型不在本地就下载（支持 hf-mirror）。"""
    d = model_dir()
    model, tokens = d / "model.int8.onnx", d / "tokens.txt"
    if model.exists() and tokens.exists():
        return d
    from huggingface_hub import snapshot_download

    endpoint = (cfg.hf_endpoint if cfg else "") or "https://hf-mirror.com"
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    print(f"[sensevoice] 下载模型到 {d}（约 230MB，首次需要一点时间）", flush=True)
    snapshot_download(repo_id=SENSEVOICE_REPO, local_dir=str(d),
                      allow_patterns=["model.int8.onnx", "tokens.txt"])
    return d


class SenseVoiceASR:
    """和 whispertype.asr.ASR 同样的 transcribe() 接口，方便上层无感切换。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._rec = None

    @property
    def loaded(self) -> bool:
        return self._rec is not None

    def load(self) -> None:
        if self._rec is not None:
            return
        import sherpa_onnx

        d = ensure_model(self.cfg)
        print(f"[sensevoice] 加载模型（CPU int8，{d}）", flush=True)
        self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(d / "model.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=self.cfg.cpu_threads or 4,
            use_itn=bool(getattr(self.cfg, "itn", True)),
            language=self.cfg.language or "zh",
        )
        print("[sensevoice] 就绪", flush=True)

    def transcribe(self, audio: np.ndarray, sr: int = TARGET_SR) -> ASRResult:
        if self._rec is None:
            self.load()
        x = np.asarray(audio, dtype=np.float32)
        if sr != TARGET_SR:  # pragma: no cover - 调用方应已重采样
            x = resample_to(x, sr, TARGET_SR)
        duration = len(x) / TARGET_SR
        t0 = time.perf_counter()
        stream = self._rec.create_stream()
        stream.accept_waveform(TARGET_SR, x)
        self._rec.decode_stream(stream)
        text = str(stream.result.text or "").strip()
        elapsed = time.perf_counter() - t0
        # 上层只用到「一整句」的结果；这里给一个等价的分段，保持接口一致
        segments = [{"start": 0.0, "end": duration, "text": text,
                     "avg_logprob": 0.0, "no_speech_prob": 0.0}] if text else []
        return ASRResult(text=text, language=self.cfg.language or "zh",
                         duration=duration, elapsed=elapsed, segments=segments)

    def transcribe_file(self, path: str | Path) -> ASRResult:
        from .audio import load_wav

        rec = load_wav(path)
        return self.transcribe(rec.samples, rec.samplerate)