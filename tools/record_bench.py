"""录 P0 基线语料：50 句，按提示用**气声**读，自动静音断句、自动存 WAV。

    python tools/record_bench.py --device BY-V20

操作：按回车开始说这一句；说完停 0.8 秒自动结束；s 跳过，r 重录，q 退出。
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispertype import audio, vad

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "bench_zh.tsv"
OUT = ROOT / "bench"


def load_manifest() -> list[tuple[str, str, str]]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sid, cat, text = line.split("\t", 2)
        rows.append((sid, cat, text))
    return rows


def record_utterance(device, sr=48000, max_seconds=10.0, silence_s=0.8) -> np.ndarray:
    q: queue.Queue = queue.Queue()
    chunks: list[np.ndarray] = []
    state = {"speech": False, "silence": 0.0, "t0": time.perf_counter()}

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(device=device, channels=1, samplerate=sr, dtype="float32",
                        blocksize=1024, callback=callback):
        time.sleep(0.4)  # 给用户一点准备时间，别把开口第一个字算进噪声
        # 用 0.5 秒里的低分位估噪声，避免一开口就把阈值抬高
        noise = []
        end = time.perf_counter() + 0.5
        while time.perf_counter() < end:
            try:
                noise.append(q.get(timeout=0.2))
            except queue.Empty:
                pass
        if noise:
            levels = np.array([np.sqrt(np.mean(c**2) + 1e-12) for c in noise])
            floor = float(np.percentile(levels, 10))
        else:
            floor = 0.002
        thr = min(max(floor * 3.0, 0.004), 0.02)

        while True:
            if time.perf_counter() - state["t0"] > max_seconds:
                break
            try:
                c = q.get(timeout=0.2)
            except queue.Empty:
                continue
            chunks.append(c)
            rms = float(np.sqrt(np.mean(c**2) + 1e-12))
            if rms > thr:
                state["speech"] = True
                state["silence"] = 0.0
            elif state["speech"]:
                state["silence"] += len(c) / sr
                if state["silence"] >= silence_s:
                    break

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def preflight(device, sr=48000) -> float | None:
    """开录前检查麦克风是否有信号；返回峰值 dBFS。"""
    try:
        x = sd.rec(int(1.0 * sr), samplerate=sr, channels=1, device=device, dtype="float32")
        sd.wait()
        return vad.peak_db(x[:, 0])
    except Exception as e:
        print(f"打不开麦克风：{e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--start", type=int, default=1, help="从第几句开始")
    ap.add_argument("--ids", default="", help="只录这些编号，逗号分隔，例如 012,021")
    ap.add_argument("--limit", type=int, default=0, help="最多录几句")
    ap.add_argument("--skip-check", action="store_true", help="跳过开录前的麦克风检查")
    args = ap.parse_args()

    device = audio.resolve_device(args.device) if args.device else None
    if not args.skip_check:
        print("检查麦克风…", flush=True)
        peak = preflight(device)
        if peak is None:
            return
        if peak < -80:
            print(f"✗ 麦克风没有信号（峰值 {peak:.1f} dBFS）。")
            print("  检查：发射器是否开机/配对、是否静音、接收器插紧、换 USB 口。")
            print("  确认有信号后重跑；急的话加 --skip-check。")
            raise SystemExit(2)
        print(f"✓ 麦克风有信号（环境峰值 {peak:.1f} dBFS）\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        rows = [r for r in rows if r[0] in wanted]
    if args.limit:
        rows = rows[: args.limit]
    print(f"共 {len(rows)} 句，用气声读；录音存到 {out}\n")

    i = max(0, args.start - 1)
    while i < len(rows):
        sid, cat, text = rows[i]
        wav = out / f"{sid}.wav"
        print(f"[{i + 1}/{len(rows)}] {cat}  {text}")
        if wav.exists():
            print(f"  已有 {wav.name}")
        try:
            cmd = input("  回车开始 / s 跳过 / r 重录 / q 退出 > ").strip().lower()
        except EOFError:
            print("\n（没有交互输入，退出）")
            break
        if cmd == "q":
            break
        if cmd == "s":
            i += 1
            continue
        if cmd == "r" and wav.exists():
            wav.unlink()

        print("  ● 开始说…（停顿 0.8 秒自动结束）", flush=True)
        x = record_utterance(device, max_seconds=args.max_seconds)
        if len(x) < int(0.3 * 48000):
            print("  ✗ 太短，没保存\n")
            continue
        audio.save_wav(wav, audio.normalize_rms(x, -20.0), 48000)
        secs = len(x) / 48000
        peak = vad.peak_db(x)
        print(f"  ✓ 已存 {wav.name}  {secs:.2f}s  峰值 {peak:.1f} dBFS")
        if peak < -32:
            print("    提示：太小声了，麦再靠近嘴边一点")
        print()
        i += 1

    done = len(list(out.glob("*.wav")))
    print(f"完成 {done}/{len(rows)} 句。")
    if done >= len(rows):
        print("下一步：python tools/benchmark.py")


if __name__ == "__main__":
    main()