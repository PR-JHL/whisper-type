"""LLM 纠错实验：SenseVoice 的识别结果再用百炼上的 DeepSeek 过一遍，看 CER 能降多少。

用法：
    python tools/exp_llm_fix.py                    # 48 句真实气声
    python tools/exp_llm_fix.py --limit 20 --model deepseek-v4.1-flash-1017
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispertype.asr import make_asr  # noqa: E402
from whispertype.audio import load_wav, prepare  # noqa: E402
from whispertype.config import Config  # noqa: E402
from whispertype.metrics import report  # noqa: E402
from whispertype.textproc import clean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "bench_zh.tsv"
BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

PROMPT = """你是中文语音识别（ASR）的后处理器。下面是某句话的 ASR 原始输出，
可能有同音字、漏字、多字、数字写法不一致或标点缺失。

要求：
1. 只输出修正后的这一句话，不要解释、不要加引号；
2. 不改变原意，不要扩写，不要编造没有的内容；
3. 常见场景：日常中文、电脑/编程/办公（如 debug、JSON、提交、日志、接口、仓库）；
4. 如果原文已经通顺，就原样输出。

ASR 原文：{text}"""


def load_key() -> str:
    key = os.environ.get("ALI_BAILIAN_API_KEY")
    if key:
        return key
    data = yaml.safe_load((Path.home() / ".dsh" / ".credentials.yaml").read_text(encoding="utf-8"))
    return data["refs"]["ALI_BAILIAN_API_KEY"]


def fix(session: requests.Session, text: str, model: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = session.post(f"{BASE}/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
        "temperature": 0,
        "max_tokens": 200,
    }, timeout=60)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return text, dt
    msg = (r.json().get("choices") or [{}])[0].get("message") or {}
    out = (msg.get("content") or "").strip().strip("“”\"'")
    return out or text, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="bench")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", default="001,002")
    ap.add_argument("--model", default="deepseek-v4.1-flash-1017")
    ap.add_argument("--json-out", default=str(ROOT / "docs" / "llm纠错.json"))
    args = ap.parse_args()

    key = load_key()
    S = requests.Session()
    S.trust_env = False
    S.headers.update({"Authorization": f"Bearer {key}", "User-Agent": "dsh"})

    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, cat, text = line.split("\t", 2)
            if sid not in {s.strip() for s in args.skip.split(",") if s.strip()}:
                rows.append((sid, cat, text))
    if args.limit:
        rows = rows[: args.limit]

    cfg = Config.load()
    asr = make_asr(cfg)
    base_e = base_n = fix_e = fix_n = 0
    latencies, improved, worsened = [], 0, 0
    items = []
    for sid, cat, ref in rows:
        p = ROOT / args.dir / f"{sid}.wav"
        if not p.exists():
            continue
        rec = load_wav(p)
        x = prepare(rec.samples, rec.samplerate, cfg)
        res = asr.transcribe(x)
        hyp = clean(res.text)
        fixed, dt = fix(S, hyp, args.model)
        fixed = clean(fixed)
        rep0, rep1 = report(ref, hyp), report(ref, fixed)
        base_e += rep0["edits"]; base_n += rep0["ref_chars"]
        fix_e += rep1["edits"]; fix_n += rep1["ref_chars"]
        latencies.append(dt)
        if rep1["edits"] < rep0["edits"]:
            improved += 1
        elif rep1["edits"] > rep0["edits"]:
            worsened += 1
        items.append({"id": sid, "category": cat, "ref": ref, "asr": hyp,
                      "fixed": fixed, "cer": rep0["cer"], "cer_fixed": rep1["cer"],
                      "latency": round(dt, 2)})
        print(f"  {sid} {rep0['cer']:5.0%} → {rep1['cer']:5.0%}  {fixed[:36]}")

    print(f"\nCER：ASR {base_e / max(1, base_n):.1%} → LLM 纠错后 {fix_e / max(1, fix_n):.1%}")
    print(f"改善 {improved} 句 / 变差 {worsened} 句 / 共 {len(items)} 句")
    if latencies:
        print(f"纠错延迟：中位 {sorted(latencies)[len(latencies) // 2]:.2f}s，"
              f"最大 {max(latencies):.2f}s")
    Path(args.json_out).write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"明细: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())