"""配置：一个 JSON 文件，路径可用 WHISPERTYPE_HOME 覆盖。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

APP_DIR = Path(os.environ.get("WHISPERTYPE_HOME", str(Path.home() / ".whispertype")))
CONFIG_PATH = APP_DIR / "config.json"
MODELS_DIR = APP_DIR / "models"


@dataclass
class Config:
    # 音频
    device: int | None = None  # sounddevice 输入设备索引；None = 系统默认
    samplerate: int | None = None  # None = 设备默认（通常是 48000）
    target_dbfs: float = -20.0  # 归一化到多少 dBFS
    vad_margin_db: float = 10.0  # 高出噪声底多少 dB 算说话
    min_utterance: float = 0.25  # 短于这个秒数丢弃

    # 识别
    engine: str = "sensevoice"  # sensevoice（默认，气声更准更快）/ whisper
    itn: bool = True  # SenseVoice 的逆文本正则：数字转阿拉伯（做题/数字场景可能要关）
    model: str = "small"  # tiny / base / small / medium / large-v3
    compute_device: str = "cpu"  # cpu / cuda（cuda 为实验性，见文档）
    compute_type: str = "int8"  # cpu:int8/float32；cuda:float16
    cpu_threads: int = 0  # 0 = 自动
    language: str = "zh"
    beam_size: int = 5
    initial_prompt: str = "普通话。"  # 短提示：能带出简体和标点，又不会在静音上被整句复读

    # 交互
    hotkey: str = "f8"
    mode: str = "hold"  # hold=按住说话；toggle=按一下开始/再按一下结束
    inject: bool = True  # 把结果送进当前前台窗口
    paste_threshold: int = 24  # 超过这么多字用剪贴板粘贴，更快
    hud: bool = True

    # 可选 LLM 后处理（修同音字/补标点）
    llm_enabled: bool = False
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_api_key_env: str = "DEEPSEEK_API_KEY"
    llm_timeout: float = 3.0

    # 模型下载（国内可设 https://hf-mirror.com）
    hf_endpoint: str = ""

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> "Config":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path | str = CONFIG_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def merged(self, **overrides) -> "Config":
        data = asdict(self)
        for k, v in overrides.items():
            if v is not None and k in data:
                data[k] = v
        return Config(**data)

    def ensure_dirs(self) -> None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def apply_env(self) -> None:
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)