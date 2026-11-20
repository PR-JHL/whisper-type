"""VAD 裁剪对气声 CER 的影响：气声起音很弱，裁首尾可能把字一起裁掉。

用法：
    python tools/exp_trim.py                 # 48 句有效真实气声
    python tools/exp_trim.py --dir bench-close
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispertype import vad  # noqa: E402
from whispertype.asr import make_asr  # noqa: E402
from whispertype.audio import TARGET_SR, highpass, load_wav, normalize_rms, resample_to  # noqa: E402
from whispertype.config import Config  # noqa: E402
from whispertype.metrics import report  # noqa: E402
from whispertype.textproc import clean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "bench_zh.tsv"


def load_rows(limit: int, skip: set[str]) -> list[tuple[str, str, str]]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, cat, text = line.split("\t", 2)
            if sid not in skip:
                rows.append((sid, cat, text))
    return rows[:limit] if limit else rows


def base(x: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    x = resample_to(x, sr, TARGET_SR)
    x = highpass(x, TARGET_SR, 60.0)
    return normalize_rms(x, cfg.target_dbfs)


VARIANTS = {
    "prepare": lambda x, sr, cfg: _prepare(x, sr, cfg),
    "no_trim": lambda x, sr, cfg: base(x, sr, cfg),
    "pad_0.3": lambda x, sr, cfg: _prepare(x, sr, cfg, pad=0.30),
    "margin_4db": lambda x, sr, cfg: _prepare(x, sr, cfg, margin_db=4.0),
    "margin_18db": lambda x, sr, cfg: _prepare(x, sr, cfg, margin_db=18.0),
}


def _prepare(x: np.ndarray, sr: int, cfg: Config, margin_db: float | None = None,
             pad: float = 0.12) -> np.ndarray:
    x = base(x, sr, cfg)
    trimmed = vad.trim(x, TARGET_SR, margin_db=cfg.vad_margin_db if margin_db is None else margin_db,
                       pad=pad)
    if len(trimmed) >= int(0.1 * TARGET_SR):
        x = trimmed
    return normalize_rms(x, cfg.target_dbfs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="bench")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", default="001,002")
    ap.add_argument("--engine", default=None, choices=["whisper", "sensevoice"])
    args = ap.parse_args()

    rows = load_rows(args.limit, {s.strip() for s in args.skip.split(",") if s.strip()})
    cfg = Config.load().merged(engine=args.engine)
    asr = make_asr(cfg)
    audio_dir = ROOT / args.dir
    print(f"{args.dir}: {len(rows)} 句 · 引擎 {cfg.engine}")
    for name, fn in VARIANTS.items():
        edits = chars = 0
        for sid, cat, ref in rows:
            p = audio_dir / f"{sid}.wav"
            if not p.exists():
                continue
            rec = load_wav(p)
            x = fn(rec.samples, rec.samplerate, cfg)
            res = asr.transcribe(x)
            rep = report(ref, clean(res.text))
            edits += rep["edits"]
            chars += rep["ref_chars"]
        print(f"  {name:12} CER {edits / max(1, chars):6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())