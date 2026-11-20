"""whispertype —— 气声输入：按住热键小声说，松开把文字送进当前窗口。

模块：
    audio     录音/重采样/归一化/WAV
    vad       能量 VAD、首尾裁剪
    asr       faster-whisper 封装
    textproc  后处理（空格/标点/复读/幻觉）
    metrics   CER 评测
    inject    Windows SendInput / 剪贴板粘贴
    hotkey    pynput 全局热键
    llm       可选 LLM 纠错
    hud/app   小窗口与主循环
"""

import os as _os

# Anaconda 的 MKL 和 ctranslate2 各自带一份 OpenMP，同进程会报
# "OMP: Error #15 ... libiomp5md.dll already initialized"。
# 让 MKL 用顺序线程，避免两份运行时打架；必须早于 numpy/scipy 导入。
_os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

__version__ = "0.1.0"