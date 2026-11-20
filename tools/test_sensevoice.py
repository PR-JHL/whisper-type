"""用 SenseVoiceSmall 跑同一批基线，和 Whisper 对比 CER。

首次运行会下载模型到 ~/.whispertype/models/sensevoice。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispertype.audio import TARGET_SR, load_wav, prepare, resample_to
from whispertype.config import Config
from whispertype.metrics import report
from whispertype.textproc import clean

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path.home() / ".whispertype" / "models" / "sensevoice"
REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


def ensure_model() -> Path:
    model = MODEL_DIR / "model.int8.onnx"
    tokens = MODEL_DIR / "tokens.txt"
    if model.exists() and tokens.exists():
        return MODEL_DIR
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"[sensevoice] 下载模型到 {MODEL_DIR} …", flush=True)
    snapshot_download(
        repo_id=REPO,
        local_dir=str(MODEL_DIR),
        allow_patterns=["model.int8.onnx", "tokens.txt"],
    )
    return MODEL_DIR


def refs() -> dict[str, str]:
    out = {}
    for line in (ROOT / "data" / "bench_zh.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, _, text = line.split("\t", 2)
            out[sid] = text
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="012,021,013,011,004,003,010,016,031,044")
    args = ap.parse_args()
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    if ids == ["all"]:
        ids = [p.stem for p in sorted((ROOT / "bench").glob("*.wav"))]

    import sherpa_onnx

    d = ensure_model()
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(d / "model.int8.onnx"),
        tokens=str(d / "tokens.txt"),
        num_threads=4,
        use_itn=True,
        language="zh",
    )
    cfg = Config()
    R = refs()
    edits = chars = 0
    for sid in ids:
        rec = load_wav(ROOT / "bench" / f"{sid}.wav")
        x = prepare(rec.samples, rec.samplerate, cfg)
        stream = recognizer.create_stream()
        stream.accept_waveform(TARGET_SR, np.asarray(x, dtype=np.float32))
        recognizer.decode_stream(stream)
        hyp = clean(stream.result.text)
        rep = report(R[sid], hyp)
        edits += rep["edits"]
        chars += rep["ref_chars"]
        print(f"{sid} CER {rep['cer']:5.0%}  {hyp}")
    print(f"\nSenseVoice 子集 CER {edits / max(1, chars):.1%}")


if __name__ == "__main__":
    main()