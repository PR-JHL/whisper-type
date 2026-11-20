# 接入 DSH 语音输入

DSH 已经自带了语音输入链路（浏览器录音 → Host `speechToText` 服务 → provider）。
我们不需要改前端，只加一个 **本地 Whisper provider**，把默认识别器从 SenseVoice 换成
`whisper-type` 的引擎。

## 链路

```
浏览器麦克风（16k PCM16 WAV）
        │  speech Remote
        ▼
ctx.speechToText  ── 默认 provider: whisper-local
        │
        ▼
@local/dsh-whisper-stt（Host 插件，独立目录，未随本仓库发布）
        │  POST /transcribe  (127.0.0.1:17321)
        ▼
python -m whispertype.server（faster-whisper，模型常驻）
        │
        ▼
文本 → 插入 DSH 输入框草稿
```

## 已安装的插件

| 项 | 值 |
|---|---|
| 包名 | `@local/dsh-whisper-stt` |
| 源码 | `E:\DSH_FairyLand\dsh-whisper-stt` |
| provider id | `whisper-local` |
| 默认语言 | `zh` |
| 本地服务 | `http://127.0.0.1:17321` |
| 模型 | `small`（CPU int8） |

插件随 DSH 启动预热，自动拉起识别服务；DSH 退出或插件禁用时结束子进程。
如果 17321 已经有服务在跑（比如你手动起的），插件会直接复用，不会重复启动。

## 验证

1. 服务健康：

```powershell
Invoke-RestMethod http://127.0.0.1:17321/health
# {"ok":true,"model":"small","compute":"cpu/int8",...}
```

2. 当前选中的 provider（插件写的诊断文件）：

```powershell
Get-Content "$env:USERPROFILE\.whispertype\plugin-status.json"
```

应看到：

```json
"selection": { "providerId": "whisper-local", "language": "zh" },
"providers": [ { "id": "sensevoice-local", ... }, { "id": "whisper-local", "phase": "ready" } ]
```

3. provider 直连测试（不经过浏览器）：

```powershell
cd ../dsh-whisper-stt
$env:ELECTRON_RUN_AS_NODE='1'
& "<DSH 安装目录>\DeepSeek Harness.exe" smoke.mjs ../whisper-type/tests/data/tts_zh.wav
```

应输出 `provider: whisper-local [...]` 和正确文本。

4. UI：刷新 DSH 页面，点输入框旁的麦克风，录一句 → 停止 → 文本进草稿。
   如果识别器还是 SenseVoice：插件详情 → 语音输入设置里把 recognizer 选成
   `Whisper 气声（本地）`。

## 改配置

编辑 `dsh-whisper-stt/cordis.patch.yml` 的 config 后重新安装：

```text
plugin_manager install_bundle target="../dsh-whisper-stt"
```

常用项：

| 字段 | 说明 |
|---|---|
| `model` | `small` / `medium` / `tiny` |
| `serverPort` | 换端口时同时改插件和防火墙设置 |
| `projectDir` | whisper-type 目录 |
| `python` | 解释器路径（默认 PATH 里的 python） |
| `autoStart` | false 时要求自己先启动服务 |

## 常见问题

| 现象 | 处理 |
|---|---|
| 麦克风按钮不可见 | 语音输入 bundle 没启用；插件页启用 Voice Input |
| `Cannot read properties of undefined (reading 'replace')` | provider 返回值契约不对：必须返回 `{ text, audioSeconds, inferenceSeconds }` 对象，不能只返回字符串 |
| 点麦克风提示需要准备模型 | 正常，这个 provider 无下载步骤；检查插件是否加载（status 文件） |
| 识别报 `whisper server HTTP 500` | 看 `whisper-type` 服务 stderr；多数是 WAV 格式或模型加载失败 |
| 首次说话很慢 | 服务预热中；等 status 里 `phase: ready` |
| 端口被占 | 改 `serverPort`，或先关掉占用进程 |
| 想切回 SenseVoice | 语音输入设置里换 provider，或把 patch 里 defaultProvider 改回 `sensevoice-local` |

## 隐私

音频只在浏览器 → DSH Host → 本机服务的链路里传输，不落盘、不出本机；
只有识别后的文本进入会话草稿。模型本地推理。