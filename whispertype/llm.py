"""可选：用 LLM 修正同音字和标点（默认关闭，失败不影响主流程）。

只依赖标准库；兼容 OpenAI 风格的 /chat/completions 接口（DeepSeek 等）。
"""

from __future__ import annotations

import json
import os
import urllib.request

from .config import Config

SYSTEM = (
    "你是中文语音输入法的后处理模块。用户用气声（小声）说话，语音识别结果可能有同音字错误、"
    "缺标点、重复词。请在不增删信息、不改原意的前提下，输出修正后的文本。"
    "不要解释，不要加引号，只输出结果本身。"
)


def available(cfg: Config) -> bool:
    return bool(os.environ.get(cfg.llm_api_key_env, "").strip())


def correct(text: str, cfg: Config) -> str:
    key = os.environ.get(cfg.llm_api_key_env, "").strip()
    if not key or not text.strip():
        return text
    url = cfg.llm_base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": cfg.llm_model,
        "temperature": 0.0,
        "max_tokens": min(1024, max(64, len(text) * 3)),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.llm_timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        out = (data["choices"][0]["message"]["content"] or "").strip().strip('"“”')
    except Exception:
        return text
    # 合理性检查：空、明显变长、或压根没变，就当没修
    if not out or len(out) > len(text) * 2 + 12:
        return text
    return out