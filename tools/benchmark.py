"""跑基线：bench/*.wav 逐句识别，算 CER、延迟、分类汇总。

    python tools/benchmark.py --model small
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispertype.asr import ASR
from whispertype.audio import load_wav, prepare
from whispertype.config import Config
from whispertype.metrics import report
from whispertype.textproc import clean

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "bench_zh.tsv"
BENCH = ROOT / "bench"


def load_manifest() -> list[tuple[str, str, str]]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, cat, text = line.split("\t", 2)
            rows.append((sid, cat, text))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--dir", default=str(BENCH), help="WAV 目录，默认 bench/")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    bench_dir = Path(args.dir)
    tag = "" if bench_dir.resolve() == BENCH.resolve() else "_" + bench_dir.name
    json_out = Path(args.json_out or (ROOT / "docs" / f"bench_result{tag}.json"))
    md_out = Path(args.md_out or (ROOT / "docs" / f"bench_report{tag}.md"))

    cfg = Config.load().merged(model=args.model)
    asr = ASR(cfg)
    rows = load_manifest()
    if args.limit:
        rows = rows[: args.limit]

    results = []
    by_cat: dict[str, list] = defaultdict(list)
    for sid, cat, ref in rows:
        wav = bench_dir / f"{sid}.wav"
        if not wav.exists():
            continue
        rec = load_wav(wav)
        x = prepare(rec.samples, rec.samplerate, cfg)
        res = asr.transcribe(x)
        hyp = clean(res.text)
        rep = report(ref, hyp)
        item = {
            "id": sid,
            "category": cat,
            "ref": ref,
            "hyp": hyp,
            "cer": rep["cer"],
            "edits": rep["edits"],
            "chars": rep["ref_chars"],
            "duration": round(res.duration, 2),
            "elapsed": round(res.elapsed, 2),
            "rtf": round(res.rtf, 2),
        }
        results.append(item)
        by_cat[cat].append(item["cer"])
        print(f"{sid} CER {rep['cer']:6.1%}  {res.elapsed:5.2f}s  {hyp}")

    if not results:
        print(f"{bench_dir} 下没有 WAV。先跑 tools/record_bench.py")
        return

    overall = sum(i["edits"] for i in results) / max(1, sum(i["chars"] for i in results))
    total_chars = sum(i["chars"] for i in results)
    elapsed = sum(i["elapsed"] for i in results)
    audio_s = sum(i["duration"] for i in results)
    lines = [
        "# 气声基线报告",
        "",
        f"- 模型：`{cfg.model}`，设备：`{cfg.compute_device}/{cfg.compute_type}`",
        f"- 句子：{len(results)}，总字数：{total_chars}",
        f"- **总体 CER：{overall:.1%}**",
        f"- 识别总耗时 {elapsed:.1f}s / 音频 {audio_s:.1f}s（RTF {elapsed / max(1e-6, audio_s):.2f}）",
        "",
        "| 类别 | 句数 | 平均 CER |",
        "|---|---|---|",
    ]
    for cat, cers in by_cat.items():
        lines.append(f"| {cat} | {len(cers)} | {sum(cers) / len(cers):.1%} |")
    lines += ["", "| 句子 | 参考 | 识别 | CER |", "|---|---|---|---|"]
    for i in results:
        lines.append(f"| {i['id']} | {i['ref']} | {i['hyp'] or '（空）'} | {i['cer']:.1%} |")

    Path(md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(md_out).write_text("\n".join(lines), encoding="utf-8")
    Path(json_out).write_text(
        json.dumps({"overall_cer": overall, "items": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n总体 CER {overall:.1%}  ->  {md_out}")


if __name__ == "__main__":
    main()