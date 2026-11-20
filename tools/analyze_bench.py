"""分析基线录音：SNR、频段能量、谱图，判断是"气声难"还是"录音差"。"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispertype import vad
from whispertype.audio import resample_to

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
OUT = ROOT / "docs"


def load(path: Path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        ch = w.getnchannels()
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return resample_to(x, sr, 16000), 16000


def band_ratios(x, sr=16000):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    total = spec.sum() + 1e-12
    bands = {"0-300": (0, 300), "300-1k": (300, 1000), "1k-4k": (1000, 4000), "4k-8k": (4000, 8000)}
    return {k: float(spec[(f >= a) & (f < b)].sum() / total) for k, (a, b) in bands.items()}


def snr_db(x, sr=16000):
    rms = vad.frame_rms(x, sr)
    db = vad.to_db(rms)
    return float(np.percentile(db, 90) - np.percentile(db, 20))


def main() -> None:
    items = json.loads((OUT / "bench_result.json").read_text(encoding="utf-8"))["items"]
    by_id = {i["id"]: i for i in items}
    rows = []
    for p in sorted(BENCH.glob("*.wav")):
        x, sr = load(p)
        rows.append(
            {
                "id": p.stem,
                "snr": snr_db(x, sr),
                "bands": band_ratios(x, sr),
                "cer": by_id.get(p.stem, {}).get("cer"),
            }
        )
    rows.sort(key=lambda r: (r["cer"] if r["cer"] is not None else 9))
    print(f"{'id':4}{'CER':>7}{'SNR':>7}{'0-300':>8}{'300-1k':>8}{'1k-4k':>8}{'4k-8k':>8}")
    for r in rows:
        b = r["bands"]
        print(
            f"{r['id']:4}{(r['cer'] or 0)*100:6.0f}%{r['snr']:6.1f}"
            f"{b['0-300']:8.2f}{b['300-1k']:8.2f}{b['1k-4k']:8.2f}{b['4k-8k']:8.2f}"
        )
    # 谱图：好、坏、静音各一
    picks = ["012", "016", "021", "001"]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.4 * len(picks)), sharex=False)
    for ax, sid in zip(axes, picks):
        p = BENCH / f"{sid}.wav"
        if not p.exists():
            continue
        x, sr = load(p)
        ax.specgram(x, NFFT=512, Fs=sr, noverlap=256, cmap="magma", vmin=-120)
        cer = by_id.get(sid, {}).get("cer")
        ax.set_title(f"{sid}  CER {cer:.0%}  SNR {snr_db(x, sr):.1f}dB" if cer is not None else sid, fontsize=10)
        ax.set_ylabel("Hz")
    axes[-1].set_xlabel("时间 / s")
    fig.tight_layout()
    fig.savefig(OUT / "bench_spectrograms.png", dpi=140)
    print("wrote", OUT / "bench_spectrograms.png")


if __name__ == "__main__":
    main()