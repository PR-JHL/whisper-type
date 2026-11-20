"""把手机录的音整理成评测目录。

手机录音（record_phone.html 生成，文件名 `<mode>-<id>.<ext>`）→ 16k 单声道 WAV →
`bench-modes-<mode>/<id>.wav`，并输出体检报告（时长/电平/是否静音）。

用法：
    python tools/import_recordings.py --src D:\\手机录音
    python tools/import_recordings.py --src 录音/ --only whisper,low
    python tools/import_recordings.py --src 录音/ --dry-run

文件名约定：`whisper-003.webm` / `low-003.m4a` / `normal-003.wav`
（mode 也可以是 `气声`/`低声`/`正常`，会自动映射）
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "bench_zh.tsv"
DOCS = ROOT / "docs"

MODE_ALIASES = {
    "whisper": "whisper", "气声": "whisper", "qisheng": "whisper", "breath": "whisper",
    "low": "low", "低声": "low", "quiet": "low",
    "normal": "normal", "正常": "normal", "voiced": "normal",
}
AUDIO_EXT = {".wav", ".webm", ".m4a", ".mp3", ".ogg", ".opus", ".aac", ".mp4", ".flac"}


def find_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for cand in (shutil.which("ffmpeg"),
                 str(ROOT.parent / "tools" / "ffmpeg" / "ffmpeg.exe"),
                 str(ROOT / "tools" / "ffmpeg" / "ffmpeg.exe")):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit("找不到 ffmpeg：--ffmpeg 指定路径，或把 ffmpeg 放进 PATH")


def load_refs() -> dict[str, str]:
    refs = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, _, text = line.split("\t", 2)
            refs[sid] = text
    return refs


def parse_name(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    if "-" not in stem:
        return None
    mode_raw, sid = stem.split("-", 1)
    mode = MODE_ALIASES.get(mode_raw.lower()) or MODE_ALIASES.get(mode_raw)
    if not mode or not sid.strip().isdigit():
        return None
    return mode, sid.strip().zfill(3)


def convert(ffmpeg: str, src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def stats(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) == 0:
        return {"duration": 0.0, "rms_dbfs": -120.0, "active": 0.0}
    win = int(sr * 0.02)
    frames = [x[i:i + win] for i in range(0, max(1, len(x) - win), win)]
    energies = np.array([float(np.mean(f ** 2)) for f in frames]) if frames else np.zeros(1)
    active = float((energies > max(1e-9, energies.max() * 0.01)).mean()) if len(energies) else 0.0
    return {"duration": round(len(x) / sr, 2),
            "rms_dbfs": round(float(10 * np.log10(float(np.mean(x ** 2)) + 1e-12)), 1),
            "active": round(active, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description="整理手机录音 → bench-modes-<mode>/")
    ap.add_argument("--src", required=True, help="手机录音所在目录（递归找）")
    ap.add_argument("--out-root", default=str(ROOT), help="输出根目录，默认项目根")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg 路径（默认找 PATH 或 ../tools/ffmpeg）")
    ap.add_argument("--only", default="", help="只处理这些模式，如 whisper,low")
    ap.add_argument("--dry-run", action="store_true", help="只报清单，不转码")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"目录不存在：{src}")
    only = {MODE_ALIASES.get(m.strip().lower(), m.strip()) for m in args.only.split(",") if m.strip()}
    refs = load_refs()
    ffmpeg = None if args.dry_run else find_ffmpeg(args.ffmpeg)

    found: dict[tuple[str, str], Path] = {}
    ignored = []
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
            continue
        parsed = parse_name(p)
        if not parsed:
            ignored.append(p.name)
            continue
        mode, sid = parsed
        if only and mode not in only:
            continue
        if sid not in refs:
            ignored.append(f"{p.name}（没有这个句子 id）")
            continue
        if (mode, sid) in found:
            print(f"  [提醒] {p.name} 与 {found[(mode, sid)].name} 重复，用后者")
        found[(mode, sid)] = p

    if not found:
        raise SystemExit(f"没找到符合命名规则的文件（<mode>-<id>.<ext>）：{src}")
    if ignored:
        print(f"忽略了 {len(ignored)} 个文件：{', '.join(ignored[:6])}" +
              (" …" if len(ignored) > 6 else ""))

    out_root = Path(args.out_root)
    records = []
    by_mode: dict[str, list[dict]] = {}
    print(f"\n{'模式':8}{'句数':>5}{'时长':>8}{'静音/过小':>10}")
    for mode in ("whisper", "low", "normal"):
        items = sorted((sid, p) for (m, sid), p in found.items() if m == mode)
        if not items:
            continue
        out_dir = out_root / f"bench-modes-{mode}"
        rows = []
        for sid, p in items:
            dst = out_dir / f"{sid}.wav"
            if args.dry_run:
                print(f"  {p.name} -> {dst.relative_to(out_root)}")
                continue
            try:
                convert(ffmpeg, p, dst)
            except subprocess.CalledProcessError as e:
                print(f"  [失败] {p.name}: {e}")
                continue
            s = stats(dst)
            s.update({"mode": mode, "id": sid, "ref": refs[sid], "source": p.name})
            rows.append(s)
            flag = "静音" if s["rms_dbfs"] < -70 else ("太小" if s["rms_dbfs"] < -45 else "")
            if flag or s["duration"] < 0.8:
                print(f"  [提醒] {mode}-{sid} {s['duration']}s {s['rms_dbfs']}dBFS {flag or '太短'}")
        if rows:
            by_mode[mode] = rows
            total = sum(r["duration"] for r in rows)
            bad = sum(1 for r in rows if r["rms_dbfs"] < -45 or r["duration"] < 0.8)
            print(f"{mode:8}{len(rows):>5}{total:>7.1f}s{bad:>10}")

    if args.dry_run:
        print("\n[dry-run] 没有转码。")
        return 0

    missing = []
    for mode in by_mode:
        got = {r["id"] for r in by_mode[mode]}
        miss = [sid for sid in refs if sid not in got]
        if miss:
            missing.append((mode, len(miss), miss[:8]))
    DOCS.mkdir(parents=True, exist_ok=True)
    manifest = {"modes": {m: len(v) for m, v in by_mode.items()},
                "missing": [{"mode": m, "count": c, "ids": ids} for m, c, ids in missing],
                "items": [r for rows in by_mode.values() for r in rows]}
    (DOCS / "recordings_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n清单: {DOCS / 'recordings_manifest.json'}")
    for mode, count, ids in missing:
        print(f"  [缺] {mode}: 还差 {count} 句（{', '.join(ids)}…）")
    print("\n下一步：")
    for mode in by_mode:
        print(f"  python tools/benchmark.py --dir bench-modes-{mode} --engine sensevoice")
    return 0


if __name__ == "__main__":
    sys.exit(main())