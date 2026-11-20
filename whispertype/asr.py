"""faster-whisper 封装：懒加载模型、固定中文参数、返回结构化结果。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 如果调用方先 import 了 numpy（Anaconda 的 MKL 会初始化 Intel OpenMP），
# 再加载 ctranslate2 就会报 OMP #15。这个开关是 Intel 官方给的兼容办法。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .audio import TARGET_SR
from .config import MODELS_DIR, Config


_DLL_HANDLES: list = []


def _add_nvidia_dll_dirs() -> None:
    """pip 装的 nvidia-cudnn-cu12 / cublas 不在 PATH 里，手动加进 DLL 搜索目录。

    注意：os.add_dll_directory 返回的句柄必须保存，否则被 GC 回收后目录就失效了。
    """
    try:
        import os
        import site

        searched = []
        for root in site.getsitepackages():
            for sub in ("nvidia/cudnn/bin", "nvidia/cublas/bin"):
                p = os.path.join(root, *sub.split("/"))
                if os.path.isdir(p) and p not in searched:
                    searched.append(p)
                    try:
                        _DLL_HANDLES.append(os.add_dll_directory(p))
                    except Exception:
                        pass
        if searched:
            os.environ["PATH"] = ";".join(searched) + ";" + os.environ.get("PATH", "")
    except Exception:
        pass


def cuda_available() -> bool:
    """能 import ctranslate2 且能找到 cuDNN 才认为 CUDA 可用。"""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False
    try:
        import os
        import site

        for root in site.getsitepackages():
            for sub in ("nvidia/cudnn/bin", "nvidia/cublas/bin"):
                p = os.path.join(root, *sub.split("/"))
                if os.path.isdir(p) and any("cudnn" in f.lower() for f in os.listdir(p)):
                    return True
    except Exception:
        pass
    return False


@dataclass
class ASRResult:
    text: str
    language: str
    duration: float  # 音频时长（秒）
    elapsed: float  # 识别耗时（秒）
    segments: list[dict] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        return self.elapsed / max(1e-6, self.duration)


def _local_snapshot(model: str) -> str | None:
    """如果模型已经下载过，直接用本地目录，避免每次都连 HuggingFace。"""
    base = MODELS_DIR / f"models--Systran--faster-whisper-{model}"
    snaps = base / "snapshots"
    if not snaps.exists():
        return None
    dirs = [p for p in snaps.iterdir() if (p / "model.bin").exists()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(dirs[0])


class ASR:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        self.cfg.ensure_dirs()
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "没装 faster-whisper，先运行：pip install faster-whisper sounddevice pynput pyperclip"
            ) from e
        local = _local_snapshot(self.cfg.model)
        target = local or self.cfg.model
        _add_nvidia_dll_dirs()
        if self.cfg.compute_device == "auto":
            # 默认不自动上 CUDA：ctranslate2 4.5 与 cuDNN 9 在部分机器上会崩，
            # CPU int8 已经比实时快，先求稳。
            attempts = [("cpu", "int8")]
        elif self.cfg.compute_device == "cuda":
            attempts = [("cuda", self.cfg.compute_type or "float16")]
        else:
            attempts = [("cpu", self.cfg.compute_type or "int8")]
        print(
            f"[asr] 加载模型 {self.cfg.model}，"
            + ("本地缓存" if local else f"首次使用会下载到 {MODELS_DIR}"),
            flush=True,
        )
        last: Exception | None = None
        for dev, ctype in attempts:
            try:
                self._model = WhisperModel(
                    target,
                    device=dev,
                    compute_type=ctype,
                    cpu_threads=self.cfg.cpu_threads,
                    download_root=str(MODELS_DIR),
                    local_files_only=bool(local),
                )
                print(f"[asr] 就绪（{dev}/{ctype}）", flush=True)
                return
            except Exception as e:  # CUDA 缺库等情况自动退回 CPU
                last = e
                print(f"[asr] {dev}/{ctype} 不可用：{str(e)[:160]}", flush=True)
        raise RuntimeError(f"模型加载失败：{last}")

    def transcribe(self, audio: np.ndarray, sr: int = TARGET_SR) -> ASRResult:
        if self._model is None:
            self.load()
        x = np.asarray(audio, dtype=np.float32)
        if sr != TARGET_SR:  # pragma: no cover - 调用方应已重采样
            from .audio import resample_to

            x = resample_to(x, sr, TARGET_SR)
        duration = len(x) / TARGET_SR
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            x,
            language=self.cfg.language or None,
            beam_size=self.cfg.beam_size,
            initial_prompt=self.cfg.initial_prompt or None,
            condition_on_previous_text=False,  # 短句输入，避免越写越偏
            vad_filter=False,  # 我们自己做了裁剪
            without_timestamps=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            temperature=[0.0, 0.2],
        )
        segs = []
        for s in segments:
            no_speech = float(getattr(s, "no_speech_prob", 0.0))
            logprob = float(getattr(s, "avg_logprob", 0.0))
            # 静音上 Whisper 容易"编"出 prompt 里的词：no_speech 高 + 置信度低，直接丢
            if no_speech > 0.6 and logprob < -1.0:
                continue
            segs.append(
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text,
                    "avg_logprob": logprob,
                    "no_speech_prob": no_speech,
                }
            )
        text = "".join(s["text"] for s in segs).strip()
        return ASRResult(
            text=text,
            language=getattr(info, "language", self.cfg.language) or "",
            duration=duration,
            elapsed=time.perf_counter() - t0,
            segments=segs,
        )

    def transcribe_file(self, path: str | Path) -> ASRResult:
        from .audio import load_wav

        rec = load_wav(path)
        x = rec.samples
        if rec.samplerate != TARGET_SR:
            from .audio import resample_to

            x = resample_to(x, rec.samplerate, TARGET_SR)
        return self.transcribe(x, TARGET_SR)

def make_asr(cfg: Config):
    """按 cfg.engine 返回识别引擎。

    sensevoice（默认）：真实气声 CER 50.1%，CPU 上还更快；
    whisper：可换模型/调 beam，作为对照保留。
    """
    if getattr(cfg, "engine", "sensevoice") == "whisper":
        return ASR(cfg)
    from .sensevoice import SenseVoiceASR

    return SenseVoiceASR(cfg)
