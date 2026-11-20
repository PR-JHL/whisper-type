"""信号处理 / 解码参数实验：同一批句子，换一版预处理，看 CER 变好还是变坏。

气声缺基频、低频能量弱、辅音能量低；whisper 的声学模型是按正常语音训练的。
这里只测**不改模型**的几种廉价手段，先看哪个方向有信号：

    base          现有 prepare()（高通 + VAD + 归一化）
    denoise       频谱减噪（估计噪声底，按帧做谱减）
    tilt          低频补偿（反预加重，把气声缺的低频补回来）
    denoise_tilt  两者叠加
    compress      动态范围压缩（把弱辅音抬起来）
    prompt        音频不变，只换 initial_prompt（告诉模型这是气声）

用法：
    python tools/exp_preprocess.py --limit 20                 # 真实录音前 20 句
    python tools/exp_preprocess.py --dirs bench --limit 48     # 全量真实录音
    python tools/exp_preprocess.py --dirs bench-synth --limit 20
    python tools/exp_preprocess.py --variants base,denoise,prompt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispertype.asr import ASR  # noqa: E402
from whispertype.audio import load_wav, normalize_rms, prepare  # noqa: E402
from whispertype.config import Config  # noqa: E402
from whispertype.metrics import report  # noqa: E402
from whispertype.textproc import clean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "bench_zh.tsv"
DOCS = ROOT / "docs"

WHISPER_PROMPT = "以下是压低声音说的气声普通话，请转写成简体中文并加标点。"


# ---------------------------------------------------------------- 预处理变体
def spectral_denoise(x: np.ndarray, sr: int, n_fft: int = 512, hop: int = 128,
                     over: float = 1.2, floor: float = 0.05) -> np.ndarray:
    """谱减：噪声底取每帧能量的低分位，逐频点减掉。"""
    win = np.hanning(n_fft)
    if len(x) < n_fft:
        return x
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(n_frames)])
    spec = np.fft.rfft(frames, axis=1)
    mag, phase = np.abs(spec), np.angle(spec)
    frames_e = (frames ** 2).mean(axis=1)
    quiet = frames_e <= np.percentile(frames_e, 15)
    noise = np.percentile(mag[quiet], 50, axis=0) if quiet.any() else np.percentile(mag, 10, axis=0)
    clean_mag = np.maximum(mag - over * noise[None, :], floor * mag)
    y = np.fft.irfft(clean_mag * np.exp(1j * phase), n=n_fft, axis=1) * win
    out = np.zeros((n_frames - 1) * hop + n_fft)
    wsum = np.zeros_like(out)
    for i in range(n_frames):
        out[i * hop:i * hop + n_fft] += y[i]
        wsum[i * hop:i * hop + n_fft] += win ** 2
    out = out / np.maximum(wsum, 1e-8)
    return out[:len(x)].astype(np.float32)


def low_boost(x: np.ndarray, sr: int, fc: float = 900.0, gain_db: float = 6.0) -> np.ndarray:
    """低频补偿：900Hz 以下平滑给 +6dB，模拟「把气声缺的低频补回来」。"""
    if len(x) < 64:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    gain = 1.0 + (10 ** (gain_db / 20.0) - 1.0) * np.clip(1.0 - f / fc, 0.0, 1.0)
    y = np.fft.irfft(X * gain, n=len(x))
    return normalize_rms(y.astype(np.float32), -20.0)


def compress(x: np.ndarray, sr: int, target_dbfs: float = -20.0, ratio: float = 0.5,
             win_s: float = 0.02) -> np.ndarray:
    """动态范围压缩：把远低于目标的帧抬起来（弱辅音），高于目标的不动。"""
    win = max(32, int(sr * win_s))
    hop = win // 2
    if len(x) < win:
        return x
    gain = np.ones(len(x), dtype=np.float32)
    for i in range(0, len(x) - win, hop):
        e = 10 * np.log10(float(np.mean(x[i:i + win] ** 2)) + 1e-12)
        if e < target_dbfs:
            g = 10 ** ((target_dbfs - e) * ratio / 20.0)
            gain[i:i + win] = np.maximum(gain[i:i + win], min(g, 8.0))
    # 平滑增益，避免咔哒声
    gain = sps.savgol_filter(gain, 101 if len(gain) > 101 else (len(gain) // 2 * 2 + 1), 2)
    y = x * gain
    peak = float(np.max(np.abs(y)))
    return (y * (0.98 / peak) if peak > 0.99 else y).astype(np.float32)


VARIANTS = {
    "base": lambda x, sr: x,
    "denoise": lambda x, sr: normalize_rms(spectral_denoise(x, sr), -20.0),
    "tilt": lambda x, sr: low_boost(x, sr),
    "tilt9": lambda x, sr: low_boost(x, sr, gain_db=9.0),
    "tilt12": lambda x, sr: low_boost(x, sr, gain_db=12.0),
    "tilt_compress": lambda x, sr: compress(low_boost(x, sr), sr),
    "denoise_tilt": lambda x, sr: low_boost(normalize_rms(spectral_denoise(x, sr), -20.0), sr),
    "compress": lambda x, sr: compress(x, sr),
    "prompt": lambda x, sr: x,
}
PROMPT_VARIANTS = {"prompt", "denoise_prompt"}


def load_manifest(limit: int = 0) -> list[tuple[str, str, str]]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, cat, text = line.split("\t", 2)
            rows.append((sid, cat, text))
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser(description="预处理变体对气声 CER 的影响")
    ap.add_argument("--dirs", nargs="+", default=["bench"], help="音频目录，默认 bench（真实气声）")
    ap.add_argument("--limit", type=int, default=20, help="每个目录跑前 N 句（0=全部）")
    ap.add_argument("--skip", default="", help="跳过的句子 id，逗号分隔（如 001,002 静音录音）")
    ap.add_argument("--variants", default="base,denoise,tilt,denoise_tilt,compress,prompt")
    ap.add_argument("--model", default=None)
    ap.add_argument("--md-out", default=str(DOCS / "信号处理_实验.md"))
    ap.add_argument("--json-out", default=str(DOCS / "信号处理_实验.json"))
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"未知变体 {v}；可选：{', '.join(VARIANTS)}")

    rows = load_manifest(args.limit)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if skip:
        rows = [r for r in rows if r[0] not in skip]
    cfg = Config.load().merged(model=args.model)
    asr = ASR(cfg)
    base_prompt = cfg.initial_prompt

    results = []
    for d in args.dirs:
        audio_dir = ROOT / d
        if not audio_dir.is_dir():
            print(f"跳过不存在的目录 {audio_dir}")
            continue
        print(f"\n=== {d}（{len(rows)} 句）===")
        for variant in variants:
            cfg.initial_prompt = WHISPER_PROMPT if variant in PROMPT_VARIANTS else base_prompt
            t0 = time.perf_counter()
            edits = chars = n = 0
            by_cat: dict[str, list[float]] = defaultdict(list)
            for sid, cat, ref in rows:
                wav = audio_dir / f"{sid}.wav"
                if not wav.exists():
                    continue
                rec = load_wav(wav)
                x = prepare(rec.samples, rec.samplerate, cfg)
                x = VARIANTS[variant](x, 16000)
                if len(x) < 0.1 * 16000:
                    continue
                res = asr.transcribe(x)
                hyp = clean(res.text)
                rep = report(ref, hyp)
                edits += rep["edits"]
                chars += rep["ref_chars"]
                n += 1
                by_cat[cat].append(rep["cer"])
            cer = edits / max(1, chars)
            results.append({"dir": d, "variant": variant, "sentences": n, "cer": round(cer, 4),
                            "categories": {k: round(sum(v) / len(v), 4) for k, v in by_cat.items()},
                            "seconds": round(time.perf_counter() - t0, 1)})
            print(f"  {variant:14} CER {cer:6.1%}  ({n} 句, {results[-1]['seconds']:.0f}s)")
    cfg.initial_prompt = base_prompt

    lines = ["# 气声信号处理实验", "",
             f"- 模型：`{cfg.model}`，设备：`{cfg.compute_device}/{cfg.compute_type}`",
             "- 每句都走 `prepare()`（16k、60Hz 高通、VAD 裁首尾、归一化 −20dBFS），再叠加变体",
             "", "| 数据 | 变体 | 句数 | CER | 日常 | DSH | 人名地名 | 数字标点 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["categories"]

        def pct(key: str) -> str:
            v = c.get(key)
            return f"{v:.1%}" if v is not None else "—"

        lines.append(f"| {r['dir']} | {r['variant']} | {r['sentences']} | **{r['cer']:.1%}** | "
                     f"{pct('日常')} | {pct('DSH')} | {pct('人名地名')} | {pct('数字标点')} |")
    Path(args.md_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n报告: {args.md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())