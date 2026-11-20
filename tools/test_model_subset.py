"""在基线子集上对比不同模型（small/medium/large）的 CER。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispertype.asr import ASR, make_asr
from whispertype.audio import load_wav, prepare
from whispertype.config import Config
from whispertype.metrics import report
from whispertype.textproc import clean

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--ids", default="012,021,013,011,004,003,010,016,031,044")
    args = ap.parse_args()
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]

    refs = {}
    for line in (ROOT / "data" / "bench_zh.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, cat, text = line.split("\t", 2)
            refs[sid] = text

    cfg = Config().merged(model=args.model)
    asr = make_asr(cfg)
    asr.load()
    total_edits = total_chars = 0
    for sid in ids:
        rec = load_wav(ROOT / "bench" / f"{sid}.wav")
        x = prepare(rec.samples, rec.samplerate, cfg)
        res = asr.transcribe(x)
        hyp = clean(res.text)
        rep = report(refs[sid], hyp)
        total_edits += rep["edits"]
        total_chars += rep["ref_chars"]
        print(f"{sid} CER {rep['cer']:5.0%}  {res.elapsed:5.1f}s  {hyp}")
    print(f"\n{args.model} 子集 CER {total_edits / max(1, total_chars):.1%}")


if __name__ == "__main__":
    main()