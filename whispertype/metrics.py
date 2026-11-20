"""评测：字错率 CER（中文按字，英文按词更合理，这里统一按字符）。"""

from __future__ import annotations

import re

from .textproc import strip_punct


def normalize(s: str) -> str:
    s = strip_punct(s or "")
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    return s


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r = normalize(ref)
    h = normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def report(ref: str, hyp: str) -> dict:
    r, h = normalize(ref), normalize(hyp)
    d = edit_distance(r, h)
    return {
        "ref_chars": len(r),
        "hyp_chars": len(h),
        "edits": d,
        "cer": d / max(1, len(r)),
        "accuracy": max(0.0, 1.0 - d / max(1, len(r))),
    }