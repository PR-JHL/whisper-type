"""在坏样本上对比预处理/提示词/VAD 变体，定位 CER 70% 的来源。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispertype.asr import ASR, make_asr
from whispertype.audio import TARGET_SR, load_wav, prepare, resample_to
from whispertype.config import Config
from whispertype.metrics import report
from whispertype.textproc import clean

ROOT = Path(__file__).resolve().parent.parent
IDS = ["016", "031", "044", "019", "009", "012", "021"]


def refs() -> dict[str, str]:
    out = {}
    for line in (ROOT / "data" / "bench_zh.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, _, text = line.split("\t", 2)
            out[sid] = text
    return out


def main() -> None:
    cfg = Config().merged(model="small")
    asr = make_asr(cfg)
    asr.load()
    R = refs()
    variants = {
        "A 当前(prompt)": dict(prompt="普通话。", raw=False),
        "B 原始(不裁剪)": dict(prompt="普通话。", raw=True),
        "C 无prompt": dict(prompt="", raw=False),
        "D 无prompt+原始": dict(prompt="", raw=True),
    }
    totals = {k: [0, 0] for k in variants}
    for sid in IDS:
        rec = load_wav(ROOT / "bench" / f"{sid}.wav")
        raw = resample_to(rec.samples, rec.samplerate, TARGET_SR)
        prep = prepare(rec.samples, rec.samplerate, cfg)
        print(f"\n{sid} 原始 {rec.seconds:.2f}s 裁剪后 {len(prep)/TARGET_SR:.2f}s | 参考：{R[sid]}")
        for name, opt in variants.items():
            asr.cfg.initial_prompt = opt["prompt"]
            x = raw if opt["raw"] else prep
            res = asr.transcribe(x)
            hyp = clean(res.text)
            rep = report(R[sid], hyp)
            totals[name][0] += rep["edits"]
            totals[name][1] += rep["ref_chars"]
            print(f"  {name:16} CER {rep['cer']:5.0%}  {hyp}")
    print("\n汇总：")
    for name, (e, c) in totals.items():
        print(f"  {name:16} CER {e / max(1, c):5.1%}")


if __name__ == "__main__":
    main()