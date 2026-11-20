"""级联实验：small 先跑，只有「没把握」的句子才用 medium 重跑。

目的：medium 在真实气声上 CER 明显更好（48.8% vs 57.5%），但 CPU 上慢 2.8 倍
（约 7s/句），交互式输入等不起。级联想在「多数句子用 small」的前提下拿到 medium 的准确率。

用法：
    python tools/exp_cascade.py --limit 22 --skip 001,002
    python tools/exp_cascade.py --limit 22 --skip 001,002 --thresholds -1.2,-1.0,-0.8,-0.6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispertype.asr import ASR  # noqa: E402
from whispertype.audio import load_wav, prepare  # noqa: E402
from whispertype.config import Config  # noqa: E402
from whispertype.metrics import report  # noqa: E402
from whispertype.textproc import clean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "bench_zh.tsv"


def load_manifest(limit: int, skip: set[str]) -> list[tuple[str, str, str]]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, cat, text = line.split("\t", 2)
            if sid not in skip:
                rows.append((sid, cat, text))
    return rows[:limit] if limit else rows


def conf_of(res) -> float:
    if not res.segments:
        return -99.0
    return sum(s["avg_logprob"] for s in res.segments) / len(res.segments)


def main() -> int:
    ap = argparse.ArgumentParser(description="small → medium 级联实验")
    ap.add_argument("--dir", default="bench")
    ap.add_argument("--limit", type=int, default=22)
    ap.add_argument("--skip", default="001,002")
    ap.add_argument("--thresholds", default="-1.4,-1.2,-1.0,-0.8,-0.6,-0.4")
    ap.add_argument("--json-out", default=str(ROOT / "docs" / "级联实验.json"))
    ap.add_argument("--md-out", default=str(ROOT / "docs" / "级联实验.md"))
    args = ap.parse_args()

    rows = load_manifest(args.limit, {s.strip() for s in args.skip.split(",") if s.strip()})
    audio_dir = ROOT / args.dir
    base_cfg = Config.load()
    small_cfg = Config().merged(model="small")
    med_cfg = Config().merged(model="medium")
    small, medium = ASR(small_cfg), ASR(med_cfg)

    records = []
    t_all = time.perf_counter()
    for sid, cat, ref in rows:
        wav = audio_dir / f"{sid}.wav"
        if not wav.exists():
            continue
        rec = load_wav(wav)
        x = prepare(rec.samples, rec.samplerate, base_cfg)
        t0 = time.perf_counter()
        rs = small.transcribe(x)
        t_small = time.perf_counter() - t0
        t0 = time.perf_counter()
        rm = medium.transcribe(x)
        t_med = time.perf_counter() - t0
        rep_s, rep_m = report(ref, clean(rs.text)), report(ref, clean(rm.text))
        records.append({
            "id": sid, "category": cat, "ref": ref,
            "small": {"hyp": clean(rs.text), "cer": rep_s["cer"], "edits": rep_s["edits"],
                      "chars": rep_s["ref_chars"], "conf": round(conf_of(rs), 2),
                      "seconds": round(t_small, 2)},
            "medium": {"hyp": clean(rm.text), "cer": rep_m["cer"], "edits": rep_m["edits"],
                       "chars": rep_m["ref_chars"], "conf": round(conf_of(rm), 2),
                       "seconds": round(t_med, 2)},
        })
        print(f"  {sid} small {rep_s['cer']:5.1%} (conf {conf_of(rs):+.2f}, {t_small:4.1f}s) | "
              f"medium {rep_m['cer']:5.1%} ({t_med:4.1f}s)", flush=True)

    def aggregate(choose) -> tuple[float, float, int]:
        edits = chars = 0
        secs = 0.0
        used_medium = 0
        for r in records:
            pick = choose(r)
            edits += pick["edits"]
            chars += pick["chars"]
            secs += pick["seconds"] + (r["medium"]["seconds"] if pick is r["medium"] else 0.0)
            if pick is r["medium"]:
                used_medium += 1
        return edits / max(1, chars), secs, used_medium

    lines = ["# small → medium 级联实验", "",
             f"- 数据：`{args.dir}` {len(records)} 句（真实气声，已跳过静音）",
             "- 规则：small 先跑；若平均 `avg_logprob` 低于阈值，则用 medium 重跑该句",
             "", "| 策略 | CER | 用 medium 的句数 | 总耗时 | 平均每句 |",
             "|---|---|---|---|---|"]
    print()
    all_small = aggregate(lambda r: r["small"])
    all_med = aggregate(lambda r: r["medium"])
    print(f"  全 small    CER {all_small[0]:6.1%}  {all_small[1]:6.1f}s")
    print(f"  全 medium   CER {all_med[0]:6.1%}  {all_med[1]:6.1f}s")
    lines.append(f"| 全 small | **{all_small[0]:.1%}** | 0/{len(records)} | {all_small[1]:.0f}s | {all_small[1] / len(records):.1f}s |")
    lines.append(f"| 全 medium | **{all_med[0]:.1%}** | {len(records)}/{len(records)} | {all_med[1]:.0f}s | {all_med[1] / len(records):.1f}s |")

    best = None
    for th in [float(t) for t in args.thresholds.split(",") if t.strip()]:
        cer, secs, used = aggregate(lambda r, th=th: r["small"] if r["small"]["conf"] >= th else r["medium"])
        mark = ""
        if best is None or cer < best[1] - 1e-9:
            best = (th, cer, secs, used)
        if cer <= all_med[0] + 0.01:
            mark = " ←"
        print(f"  conf<{th:+.1f}  CER {cer:6.1%}  用 medium {used:2}/{len(records)}  {secs:5.1f}s{mark}")
        lines.append(f"| 级联 conf<{th:+.1f} | {cer:.1%} | {used}/{len(records)} | {secs:.0f}s | {secs / len(records):.1f}s |")

    lines += ["", "## 怎么看", "",
              f"- small 与 medium 的准确率差 {all_small[0] - all_med[0]:.1%}，这是换模型能拿到的大头；",
              "- 级联能不能占到便宜，取决于 small 的置信度是否真能挑出「它答错的句子」。"
              "如果挑得准，就能用很小的延迟代价换到接近 medium 的准确率；挑不准就白花时间。",
              "", "## 逐句明细", "",
              "| id | small CER | conf | medium CER | 谁更好 |", "|---|---|---|---|---|"]
    for r in records:
        better = "medium" if r["medium"]["edits"] < r["small"]["edits"] else ("small" if r["small"]["edits"] < r["medium"]["edits"] else "平")
        lines.append(f"| {r['id']} | {r['small']['cer']:.0%} | {r['small']['conf']:+.2f} | "
                     f"{r['medium']['cer']:.0%} | {better} |")

    Path(args.md_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.json_out).write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n总耗时 {time.perf_counter() - t_all:.0f}s · 报告 {args.md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())