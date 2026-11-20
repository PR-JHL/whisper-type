# 气声输入 · whispertype

按住一个热键，用**气声（压低声）**说一句话，松开后文字自动出现在当前窗口。
本地识别（faster-whisper），不联网、不上传音频。

> 现状：软件链路已经跑通——内置中文测试音频的识别 CER = 0%，CPU 上 6.3s 音频约 1.7s 出字。
> **你插的 BY-V20 领夹麦目前输出全零**（WASAPI 端点是数字静音），先按下面第 0 步检查硬件。

运行 `python -m whispertype run` 会弹出一个小窗：红点=录音、蓝=识别、绿=已输入。

## 0. 先确认麦克风真的在出声

```bash
python -m whispertype devices          # 看设备列表，* 是默认
python -m whispertype doctor --record 3 --asr
```

`doctor` 会打印默认麦克风、Windows 音量/静音、1 秒采样电平，并用内置音频跑一次识别。

如果你看到 **峰值 ≈ -180 dBFS / -90 dBFS**，说明系统收到的是静音，不是软件问题：

- 无线领夹麦：发射器长按开机、看指示灯是否常亮；重新配对；确认没按静音键；充电。
- 有线领夹麦：确认 3.5mm 插到底；台式机要插**粉色麦克风口**，笔记本要 **TRRS 四极**或 USB 声卡。
- Windows 11：设置 → 隐私和安全性 → 麦克风 → 打开「允许桌面应用访问麦克风」。
- 设备选错：`--device BY-V20` 或 `--device 11`（WASAPI 通常延迟最低）。

能听到自己声音后，再继续。

## 1. 安装

```bash
cd whisper-type
pip install -r requirements.txt
```

> `ctranslate2` 锁在 4.5.0：4.9.0 在这台机器的 Anaconda 环境里加载模型会直接崩溃。
> `pip install faster-whisper` 自动带的版本可能不对，按 requirements 装。

首次运行会下载模型到 `~/.whispertype/models`（small 约 486MB）。国内网络慢可以设镜像：

```bash
set HF_ENDPOINT=https://hf-mirror.com
```

## 2. 用

```bash
python -m whispertype run
```

- 默认 **按住 F8 说话，松开识别**；识别完文字直接打进当前窗口。
- 小窗可以拖到角落；`Esc` 或点「退出」结束。
- 想改成按一下开始、再按一下结束：`--toggle`。
- 不想打字、只看识别结果：`--no-inject`。
- 热键冲突：`--hotkey f9` / `caps_lock` / `ctrl_r` / 单个字母。

第一次按下会等模型加载（几秒）。之后每次：录音 → 识别 → 注入，2 秒的话大约 1 秒内出字。

### F8 没反应？

先跑按键体检，按一下你想用的键：

```bash
python -m whispertype keytest
```

它会打印这个键**实际被系统识别成什么**：

| keytest 输出 | 含义 / 处理 |
|---|---|
| `f8` | 键没问题。确认程序在跑（小窗/控制台还在）、模型已就绪（状态从「加载中」变成「待机」） |
| `media_previous` / `media_play_pause` 等 | 键盘把 F8 当媒体键了：按 **Fn+F8**，或者换键 |
| 什么都不打印 | 全局钩子被安全软件/游戏反作弊拦了；换 `--hotkey caps_lock` 或点小窗的「按住说话」 |
| `left` / `x1` / `x2` | 鼠标键也能当热键：`--hotkey mouse4` |

换键不用改代码：

```bash
python -m whispertype run --hotkey caps_lock
python -m whispertype run --hotkey mouse4
```

小窗上还有「按住说话」按钮，完全不依赖热键。每次状态都会写进
`C:\Users\<你>\.whispertype\run.log`，出问题先看最后几行。

## 3. 接进 DSH（已完成）

DSH 自带语音输入链路，我们只补了一个本地 provider `whisper-local`，
并把默认识别器切了过去。浏览器麦克风 → Host → 本地 Whisper 服务 → 输入框草稿。

- 插件源码：`dsh-whisper-stt`（DSH 客户端插件，独立目录，不随本仓库发布）
- 本地服务：`python -m whispertype.server --port 17321`（插件自动拉起）
- 验证：`~/.whispertype/plugin-status.json` 里 `selection.providerId` 应为 `whisper-local`

细节、改配置和排错见 [docs/DSH接入.md](docs/DSH接入.md)。

**P0 基线已经跑出来了**：50 句真实气声，CER **70.8%**（见 `docs/bench_report.md`）——
气声对 whisper small 的损失很大。合成气声与真实气声的对照、以及下一步建议见
[docs/合成气声_评测.md](docs/合成气声_评测.md)。

不用等录音也能跑回归：用百炼的 TTS 造「气声」，当天就能测。

```bash
python tools/synth_whisper.py --out bench-synth   # 50 句，约 2 分钟（需要百炼 key）
python tools/benchmark.py --dir bench-synth       # 同一套评测脚本
```

合成气声 CER 18.2% / 真实气声 69.6%，四类难度排序完全一致 → 合成集能当**相对比较**的测试床，
但不能拿它的绝对 CER 当验收线。

## 4. 模型与速度

| 模型 | 大小 | CPU 速度（RTF，越小越快） | 中文效果 |
|---|---|---|---|
| tiny | 78MB | ~0.12 | 安静环境够用，气声容易错 |
| base | 145MB | ~0.2 | 介于两者之间 |
| **small（默认）** | 486MB | **0.27–0.6** | 气声推荐起点 |
| medium | 1.5GB | ~1.2 | 更准，但 CPU 上慢于实时 |

RTF = 识别耗时 / 音频时长。用 `--model tiny` 试速度，`--model medium` 试准确率。

有 NVIDIA 显卡可以试 `--compute-device cuda`，但当前 ctranslate2 4.5 + cuDNN 9 的组合
在部分机器上会崩（本机就是），**不建议默认开**。

## 5. 针对气声的调参

气声没有声带振动，低频少、电平比正常说话低 15–25dB，所以：

1. **麦克风离嘴 3–5cm**（领夹夹在领口），别放桌上；
2. Windows 输入音量 80–100，关掉「噪声抑制 / 音频增强」（它们会把气声当噪声）；
3. 说话比平时**稍慢、字与字之间别拖**，一句 2–4 秒最顺；
4. 识别结果不对时，先看 `doctor` 里的峰值：低于 -30dBFS 就是太小声；
5. 可选 LLM 纠错（同音字/标点）：

```bash
set DEEPSEEK_API_KEY=sk-...
python -m whispertype run --llm
```

## 6. 配置

配置文件：`C:\Users\<你>\.whispertype\config.json`

```bash
python -m whispertype config                    # 查看
python -m whispertype config set model medium
python -m whispertype config set hotkey f9
python -m whispertype config set mode toggle
```

常用项：`device`、`model`、`hotkey`、`mode`、`inject`、`paste_threshold`、`llm_enabled`。

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `OMP: Error #15 libiomp5md.dll already initialized` | 包已在 `__init__` 里设 `MKL_THREADING_LAYER=SEQUENTIAL`；若仍出现，手动设这个环境变量 |
| 加载模型时进程直接退出（无 traceback） | ctranslate2 版本问题，确认是 4.5.0 |
| 模型下载 SSL/连接失败 | 重试，或设 `HF_ENDPOINT=https://hf-mirror.com` |
| 热键没反应 | 游戏/输入法占用了 F8；换 `--hotkey f9`；以普通用户运行 |
| 文字没进窗口 | 当前窗口不接受模拟键盘（管理员权限窗口）；用 `--no-inject` 先看识别结果 |
| 识别出「谢谢观看」 | 静音幻觉，已过滤；仍有就调低 `vad` 或换模型 |

## 8. 结构

```
whispertype/
  audio.py     录音、重采样、归一化、WAV
  vad.py       能量 VAD、首尾裁剪
  asr.py       faster-whisper 封装（懒加载、本地缓存、CPU/GPU）
  textproc.py  去空格、修标点、去复读、过滤幻觉
  metrics.py   CER 评测
  inject.py    SendInput / 剪贴板粘贴
  hotkey.py    pynput 全局热键
  llm.py       可选 LLM 纠错
  app.py       Controller + HUD
  cli.py       命令行
tests/         29 个单元测试（+1 个需模型的集成测试，默认跳过）
tools/         麦克风回环测试、调参、合成气声（synth_whisper.py）、评测（benchmark.py）
```

`bench-synth*/` 是合成气声的产物（可随时重生成，不用提交）。

## 9. 隐私

音频只在内存里处理，默认不落盘；只有 `mic-check` / `transcribe` 会写 WAV。
模型本地推理，不需要网络（下载完以后）。LLM 纠错是唯一的外发路径，默认关闭。

**仓库里没有音频。** `bench*/`（个人录音）和由录音派生的语谱图都在 `.gitignore` 里，
`tools/synth_whisper.py` 造出来的合成气声也是本地生成、可重跑，不需要提交。

## 10. 还没做的

- 连续模式（不用按键，VAD 自动断句）——目前是按住说话，避免误触发；
- 流式识别（边说边出字）——faster-whisper 可以分段，但延迟和稳定性要调；
- 把结果接到 DSH（MCP server 或插件）——需要的话再加。