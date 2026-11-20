"""用真实气声录音校准合成气声：把「TTS 普通语句 → 气声化」的结果拉近真实录音。

昨天做的合成气声 CER 11.5%（SenseVoice），真实气声 48–50%——太干净了，不能当真实基线。
差距主要来自三件事，这里逐条对齐（都用 `bench/` 里你自己的真实录音当参考）：

1. **频谱形状（LTAS）**：真实录音经过手机/领夹麦，频响不是平的；把合成音频的长期平均谱
   向真实录音的谱形靠（1/3 倍频程平滑后再修，避免过拟合）；
2. **底噪**：真实录音里有环境噪声（空调、键盘、远处人声），合成音频是数字静音；
   从真实录音的停顿段估噪声谱和信噪比，按同样的 SNR 混进合成音频；
3. **动态**：真实说话的音量起伏比 TTS 大；用真实录音的 RMS 分布做轻微压缩/扩张。

用法：
    python tools/synth_calibrated.py --real bench --out bench-synth-cal
    python tools/synth_calibrated.py --real bench --out bench-synth-cal --limit 10 --force
    python tools/synth_calibrated.py --real bench --out bench-synth-cal --no-noise   # 只修频谱

评测：
    python tools/benchmark.py --dir bench-synth-cal --engine sensevoice
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from synth_whisper import (  # noqa: E402
    SR, TTS, WHISPER_INSTRUCTION, acoustic, load_key, load_manifest, save_wav, whisperize,
)
from whispertype.audio import load_wav, resample_to  # noqa: E402

TARGET_SR = 16000


# ---------------------------------------------------------------- 真实录音统计
def _frames(x: np.ndarray, sr: int, win_s: float = 0.032, hop_s: float = 0.010):
    win, hop = int(sr * win_s), int(sr * hop_s)
    if len(x) < win:
        return np.zeros((0, win)), np.zeros(0)
    idx = np.arange(0, len(x) - win, hop)
    frames = np.stack([x[i:i + win] for i in idx])
    return frames, (frames ** 2).mean(axis=1)


def _smooth_spectrum(power: np.ndarray, bins_per_octave: int = 3) -> np.ndarray:
    """1/3 倍频程平滑（用平均池化的近似），避免 EQ 把噪声一起学进去。"""
    n = len(power)
    width = max(3, n // 128)
    kernel = np.ones(width) / width
    return np.convolve(power, kernel, mode="same")


def analyze_real(real_dir: Path, max_files: int = 0) -> dict:
    """从真实录音里估计：LTAS、噪声谱、SNR、语音 RMS 中位数。"""
    files = sorted(real_dir.glob("*.wav"))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"{real_dir} 里没有 wav")
    n_fft = 512
    speech_pow, noise_pow, snrs, speech_rms = [], [], [], []
    for f in files:
        rec = load_wav(f)
        x = resample_to(rec.samples, rec.samplerate, TARGET_SR)
        frames, energy = _frames(x, TARGET_SR)
        if len(frames) == 0 or energy.max() <= 0:
            continue
        spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), n=n_fft, axis=1)) ** 2
        active = energy > max(1e-10, energy.max() * 0.02)
        quiet = energy <= np.percentile(energy, 12)
        if active.any():
            speech_pow.append(spec[active].mean(axis=0))
            speech_rms.append(float(np.sqrt(energy[active].mean())))
        if quiet.any():
            noise_pow.append(spec[quiet].mean(axis=0))
            n_rms = float(np.sqrt(energy[quiet].mean()))
            if speech_rms:
                snrs.append(max(1.0, speech_rms[-1] / max(1e-6, n_rms)))
    if not speech_pow:
        raise SystemExit("真实录音里没找到有效语音")
    ltas = np.mean(speech_pow, axis=0)
    noise = np.mean(noise_pow, axis=0) if noise_pow else np.zeros_like(ltas)
    snr = float(np.median(snrs)) if snrs else 20.0
    print(f"[calib] 参考录音 {len(files)} 条 · SNR 中位 {20 * np.log10(snr):.1f}dB · "
          f"语音 RMS 中位 {20 * np.log10(np.median(speech_rms)):.1f}dBFS")
    return {"ltas": ltas, "noise": noise, "snr": snr,
            "speech_rms": float(np.median(speech_rms)), "n_fft": n_fft}


def calibrate(x: np.ndarray, sr: int, calib: dict, use_noise: bool = True) -> np.ndarray:
    """把一段合成气声（24k）拉到真实录音的频谱/噪声/动态。"""
    if sr != TARGET_SR:
        x = resample_to(x, sr, TARGET_SR)
        sr = TARGET_SR
    n_fft = calib["n_fft"]
    ltas = _smooth_spectrum(calib["ltas"])
    frames, energy = _frames(x, sr)
    if len(frames) == 0:
        return x
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), n=n_fft, axis=1)) ** 2
    active = energy > max(1e-10, energy.max() * 0.02)
    cur = _smooth_spectrum(spec[active].mean(axis=0)) if active.any() else _smooth_spectrum(spec.mean(axis=0))
    # 1) LTAS 匹配：增益限制在 ±12dB，避免把静音段也一起放大
    gain = np.clip(np.sqrt(ltas / np.maximum(cur, 1e-12)), 10 ** (-12 / 20), 10 ** (12 / 20))
    X = np.fft.rfft(x)
    # 增益表是按 n_fft 算的，映射到整段声音的频点上
    gain_full = np.interp(np.linspace(0, len(gain) - 1, len(X)),
                          np.arange(len(gain)), gain)
    x = np.fft.irfft(X * gain_full, n=len(x))
    # 2) 加底噪：按真实 SNR 混入真实噪声谱（随机相位）
    if use_noise and calib["noise"].any():
        speech_rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
        noise_rms_target = speech_rms / max(1.0, calib["snr"])
        rng = np.random.default_rng(7)
        noise_frames = []
        hop = len(x) // max(1, len(x) // 256) or 256
        for i in range(0, len(x) - n_fft, hop):
            phase = rng.uniform(-np.pi, np.pi, n_fft // 2 + 1)
            frame = np.fft.irfft(np.sqrt(calib["noise"] / 2.0) * np.exp(1j * phase), n=n_fft)
            noise_frames.append(frame)
        if noise_frames:
            noise = np.zeros(len(x))
            for i, fr in enumerate(noise_frames):
                a = i * hop
                b = min(a + n_fft, len(noise))
                noise[a:b] += fr[: b - a]
            cur_rms = float(np.sqrt(np.mean(noise ** 2)) + 1e-12)
            x = x + noise * (noise_rms_target / cur_rms)
    # 3) 电平拉到真实录音的语音 RMS（prepare() 之后还会归一化，这里只是别差太远）
    target = calib["speech_rms"]
    cur_rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    x = x * (target / cur_rms)
    peak = float(np.max(np.abs(x)))
    if peak > 0.95:
        x = x * (0.95 / peak)
    return x.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="用真实录音校准合成气声")
    ap.add_argument("--real", default=str(ROOT / "bench"), help="真实录音目录（默认 bench/）")
    ap.add_argument("--out", default=str(ROOT / "bench-synth-cal"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-noise", action="store_true", help="只做频谱匹配，不加底噪")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-real", type=int, default=0, help="只用前 N 条真实录音做统计")
    args = ap.parse_args()

    real_dir = Path(args.real)
    out_dir = Path(args.out)
    calib = analyze_real(real_dir, args.max_real)
    tts = TTS(load_key())
    rows = load_manifest(args.limit)
    print(f"[calib] 生成 {len(rows)} 句 → {out_dir}（加底噪={not args.no_noise}）")

    meta = []
    for sid, cat, text in rows:
        wav_path = out_dir / f"{sid}.wav"
        if wav_path.exists() and not args.force:
            print(f"  {sid} 已存在，跳过")
            continue
        try:
            audio = tts.say(text, instruction=WHISPER_INSTRUCTION)
            audio = whisperize(audio, SR, seed=int(sid) if sid.isdigit() else 0)
            audio = calibrate(audio, SR, calib, use_noise=not args.no_noise)
            save_wav(wav_path, audio, TARGET_SR)
            m = acoustic(audio, TARGET_SR)
            meta.append({"id": sid, "category": cat, "ref": text, **m})
            print(f"  {sid} {m['duration']:5.2f}s HNR {m['hnr_db']}dB voiced {m['voiced_ratio']:.0%} "
                  f"rms {m['rms_dbfs']}dBFS")
        except Exception as e:  # noqa: BLE001
            print(f"  {sid} 失败：{type(e).__name__}: {e}")
    if meta:
        import json
        (out_dir / "synth_meta.json").write_text(
            json.dumps({"calibrated_from": str(real_dir), "items": meta},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n{len(meta)} 句完成 · 下一步：python tools/benchmark.py --dir {args.out} --engine sensevoice")
    return 0


if __name__ == "__main__":
    sys.exit(main())