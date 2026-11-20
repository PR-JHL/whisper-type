"""把多个目录的评测结果汇总成一张表：模式 × 类别 × 引擎。

用法：
    python tools/compare_modes.py                       # 自动找 docs/bench_result_*.json
    python tools/compare_modes.py --glob "bench-modes-*"
    python tools/compare_modes.py --md docs/模式对比.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def summarize(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        return None
    cats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i in items:
        c = i.get("category", "?")
        cats[c][0] += int(i.get("edits", 0))
        cats[c][1] += int(i.get("chars", 0))
    e = sum(v[0] for v in cats.values())
    n = sum(v[1] for v in cats.values())
    secs = sum(float(i.get("elapsed", 0.0)) for i in items)
    dur = sum(float(i.get("duration", 0.0)) for i in items)
    return {"items": len(items), "cer": e / max(1, n), "seconds": secs, "audio": dur,
            "categories": {k: v[0] / max(1, v[1]) for k, v in cats.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总评测结果")
    ap.add_argument("--glob", default="bench_result_*.json", help="docs/ 下的文件名通配")
    ap.add_argument("--md", default=str(DOCS / "模式对比.md"))
    args = ap.parse_args()

    rows = []
    for p in sorted(DOCS.glob(args.glob)):
        s = summarize(p)
        if s:
            rows.append((p.stem.replace("bench_result_", ""), s))
    if not rows:
        print(f"docs/ 下没有匹配 {args.glob} 的结果；先跑 tools/benchmark.py")
        return 1

    all_cats = sorted({c for _, s in rows for c in s["categories"]})
    print(f"{'结果':36}{'句数':>5}{'CER':>7}{'RTF':>7}  " + "  ".join(c for c in all_cats))
    lines = ["| 结果 | 句数 | CER | RTF | " + " | ".join(all_cats) + " |",
             "|---|---|---|---|" + "---|" * len(all_cats)]
    for name, s in rows:
        rtf = s["seconds"] / max(1e-6, s["audio"])
        cat_txt = "  ".join(f"{s['categories'].get(c, float('nan')):.0%}" for c in all_cats)
        print(f"{name:36}{s['items']:>5}{s['cer']:>7.1%}{rtf:>7.2f}  {cat_txt}")
        lines.append(f"| {name} | {s['items']} | **{s['cer']:.1%}** | {rtf:.2f} | " +
                     " | ".join(f"{s['categories'].get(c, float('nan')):.1%}" for c in all_cats) + " |")

    md = Path(args.md)
    md.write_text("# 评测结果汇总\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n汇总: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())