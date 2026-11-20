"""命令行入口：python -m whispertype <command>"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import audio, vad
from .asr import ASR, make_asr
from .config import CONFIG_PATH, Config
from .metrics import report
from .textproc import clean


def _cfg(args) -> Config:
    cfg = Config.load()
    device = None
    if getattr(args, "device", None) not in (None, ""):
        device = audio.resolve_device(args.device)
    overrides = {
        "device": device,
        "model": getattr(args, "model", None),
        "language": getattr(args, "language", None),
        "hotkey": getattr(args, "hotkey", None),
        "compute_device": getattr(args, "compute_device", None),
        "compute_type": getattr(args, "compute_type", None),
        "mode": "toggle" if getattr(args, "toggle", False) else None,
        "inject": False if getattr(args, "no_inject", False) else None,
        "hud": False if getattr(args, "no_hud", False) else None,
        "llm_enabled": True if getattr(args, "llm", False) else None,
    }
    return cfg.merged(**overrides)


def cmd_doctor(args) -> None:
    import platform

    print(f"Python      {platform.python_version()} ({sys.executable})")
    for mod in ("faster_whisper", "ctranslate2", "sounddevice", "pynput"):
        try:
            m = __import__(mod)
            print(f"{mod:<12}{getattr(m, '__version__', 'ok')}")
        except Exception as e:
            print(f"{mod:<12}缺失（{e}）")

    from .config import MODELS_DIR

    snaps = []
    if MODELS_DIR.exists():
        for p in MODELS_DIR.glob("models--*"):
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            snaps.append(f"{p.name.replace('models--Systran--faster-whisper-', '')} ({size/1e6:.0f}MB)")
    print(f"模型缓存    {MODELS_DIR}: {', '.join(snaps) if snaps else '空（首次运行会下载）'}")

    cfg = Config.load()
    print(f"配置        {CONFIG_PATH}  model={cfg.model} hotkey={cfg.hotkey} mode={cfg.mode}")

    try:
        default = audio.default_input_index()
        devices = audio.list_input_devices()
        d = next((x for x in devices if x["index"] == default), None)
        print(f"默认麦克风  [{default}] {d['name'] if d else '?'}")
    except Exception as e:
        print(f"枚举设备失败：{e}")

    try:
        from ctypes import cast, POINTER

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        mic = AudioUtilities.GetMicrophone()
        vol = cast(mic.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None), POINTER(IAudioEndpointVolume))
        print(f"麦克风音量  {vol.GetMasterVolumeLevelScalar()*100:.0f}%  静音={bool(vol.GetMute())}")
        spk = AudioUtilities.GetSpeakers()
        print(f"扬声器      {spk.FriendlyName} {spk.volume_percent:.0f}%")
    except Exception as e:
        print(f"音量读取失败（不影响使用）：{e}")

    if args.record > 0:
        try:
            rec = audio.Recorder(cfg.device, cfg.samplerate)
            print(f"采样 {args.record:.1f}s 看看当前电平（不用说话）……", flush=True)
            rec.start()
            time.sleep(args.record)
            r = rec.stop()
            print(
                f"  峰值 {vad.peak_db(r.samples):.1f} dBFS，"
                f"噪声底 {vad.noise_floor_db(r.samples, r.samplerate):.1f} dBFS"
            )
            if vad.peak_db(r.samples) < -80:
                print("  ⚠ 接近全静音：检查麦克风开关/配对/静音键；也可能是 Windows 隐私里禁用了桌面应用")
        except Exception as e:
            print(f"采样失败：{e}")

    try:
        from .hotkey import parse_key

        parse_key(cfg.hotkey)
        print(f"热键        {cfg.hotkey} 可用")
    except Exception as e:
        print(f"热键不可用：{e}")

    if args.asr:
        sample = Path(__file__).resolve().parent.parent / "tests" / "data" / "tts_zh.wav"
        if not sample.exists():
            print("没有内置测试音频，跳过 ASR 自检")
            return
        cfg2 = cfg.merged(device=cfg.device)
        asr = make_asr(cfg2)
        res = asr.transcribe_file(sample)
        print(f"ASR 自检    {clean(res.text)}")
        print(f"            {res.duration:.1f}s 音频 / {res.elapsed:.2f}s 识别 (RTF {res.rtf:.2f})")


def cmd_keytest(args) -> None:
    """按任意键/鼠标键，打印它实际被系统识别成什么。"""
    import time as _time

    from pynput import keyboard, mouse

    from .hotkey import suggest

    print("按下你想当热键的那个键（F8、CapsLock、右Ctrl、鼠标侧键…），")
    print("这里会打印它的真实名字和建议写法；Ctrl+C 退出。\n", flush=True)

    seen: list[str] = []

    def on_press(k):
        s = suggest(k)
        line = f"按下   {s:<18} 建议写法：{s}"
        if s.startswith("media_"):
            line += "   ← 键盘把它当媒体键了，试试 Fn+那个键，或换 CapsLock"
        print(line, flush=True)
        seen.append(s)

    def on_release(k):
        pass

    def on_click(x, y, button, pressed):
        if pressed:
            b = str(button).replace("Button.", "")
            alias = {"x1": "mouse4", "x2": "mouse5"}.get(b, b)
            print(f"鼠标   {alias:<18} 建议写法：{alias}", flush=True)
            seen.append(alias)

    kl = keyboard.Listener(on_press=on_press, on_release=on_release)
    ml = mouse.Listener(on_click=on_click)
    kl.start()
    ml.start()
    try:
        while True:
            _time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        kl.stop()
        ml.stop()

    if not seen:
        print("什么都没收到：热键钩子可能被安全软件/游戏反作弊拦了。")
    print("\n用法示例：python -m whispertype run --hotkey " + (seen[-1] if seen else "caps_lock"))


def cmd_devices(args) -> None:
    try:
        devices = audio.list_input_devices()
    except Exception as e:
        print(f"枚举设备失败：{e}")
        return
    default = audio.default_input_index()
    print(f"{'':2}{'#':>3}  {'设备':<42}{'通道':>4}  {'采样率':>7}  hostapi")
    for d in devices:
        mark = " *" if d["index"] == default else "  "
        print(
            f"{mark}{d['index']:>3}  {d['name'][:40]:<42}{d['channels']:>4}"
            f"  {d['samplerate']:>7}  {d['hostapi']}"
        )
    print("\n* = 系统默认。用 --device 序号 或 --device 名称片段 指定。")


def cmd_mic_check(args) -> None:
    cfg = _cfg(args)
    try:
        rec = audio.Recorder(cfg.device, cfg.samplerate)
        print(f"录音 {args.seconds:.0f} 秒，现在开始用**气声**说话……", flush=True)
        rec.start()
        time.sleep(args.seconds)
        r = rec.stop()
    except Exception as e:
        print(f"录音失败：{e}")
        return
    if len(r.samples) == 0:
        print("没录到数据，检查设备序号。")
        return

    x, sr = r.samples, r.samplerate
    floor = vad.noise_floor_db(x, sr)
    rms = vad.rms_db(x)
    peak = vad.peak_db(x)
    ratio = vad.speech_ratio(x, sr)
    clip = float((abs(x) > 0.985).mean())

    print()
    print(f"时长      {r.seconds:.2f}s @ {sr} Hz")
    print(f"噪声底    {floor:6.1f} dBFS   （越负越好，< -55 算安静）")
    print(f"整体 RMS  {rms:6.1f} dBFS")
    print(f"峰值      {peak:6.1f} dBFS   （气声建议 -18 ~ -6）")
    print(f"说话占比  {ratio:6.0%}")
    print(f"削波比例  {clip:6.1%}")

    advice = []
    if peak < -25:
        advice.append("峰值太低：麦克风离嘴太远，或系统输入音量太小（调到 80–100）。")
    if clip > 0.001:
        advice.append("有削波：把系统输入音量调低 10–20。")
    if floor > -45:
        advice.append("噪声底偏高：开风扇/空调/机箱风扇？或增益太大。")
    if ratio < 0.15:
        advice.append("说话占比低：可能没在对准麦克风，或气声太小。")
    if not advice:
        advice.append("电平看着不错，可以直接跑 `whispertype run`。")

    out = Path(args.out or "mic-check.wav")
    audio.save_wav(out, audio.normalize_rms(x, -20.0), sr)
    print(f"\n已保存：{out.resolve()}（已归一化到 -20 dBFS，可回放确认音质）")
    for a in advice:
        print("建议：" + a)

    if args.asr:
        cfg2 = cfg.merged(model=args.model or cfg.model)
        asr = make_asr(cfg2)
        res = asr.transcribe(audio.prepare(x, sr, cfg2))
        print(f"\n识别结果：{clean(res.text) or '（空）'}")
        print(f"耗时 {res.elapsed:.2f}s / 音频 {res.duration:.1f}s (RTF {res.rtf:.2f})")


def cmd_transcribe(args) -> None:
    cfg = _cfg(args)
    if args.wav:
        rec = audio.load_wav(args.wav)
        x = audio.prepare(rec.samples, rec.samplerate, cfg)
    else:
        rec = audio.Recorder(cfg.device, cfg.samplerate)
        print(f"录音 {args.record:.0f} 秒……", flush=True)
        rec.start()
        time.sleep(args.record)
        r = rec.stop()
        x = audio.prepare(r.samples, r.samplerate, cfg)

    asr = make_asr(cfg)
    res = asr.transcribe(x)
    text = clean(res.text)
    print(f"\n识别：{text or '（空）'}")
    print(f"音频 {res.duration:.2f}s · 识别 {res.elapsed:.2f}s · RTF {res.rtf:.2f} · 语言 {res.language}")
    if args.ref:
        r = report(args.ref, text)
        print(
            f"参考 {r['ref_chars']} 字，识别 {r['hyp_chars']} 字，"
            f"编辑 {r['edits']}，CER {r['cer']:.1%}"
        )


def cmd_run(args) -> None:
    from . import app

    cfg = _cfg(args)
    cfg.apply_env()
    asr = make_asr(cfg)
    app.run(cfg, asr, hud=cfg.hud)


def cmd_config(args) -> None:
    cfg = Config.load()
    if args.action == "set":
        if len(args.pair) != 2:
            print("用法：whispertype config set <key> <value>")
            return
        k, v = args.pair
        if not hasattr(cfg, k):
            print(f"未知配置项：{k}")
            return
        cur = getattr(cfg, k)
        try:
            if isinstance(cur, bool):
                val = v.lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int) and not isinstance(cur, bool):
                val = int(v)
            elif isinstance(cur, float):
                val = float(v)
            else:
                val = v
        except ValueError:
            print(f"值 {v!r} 类型不对")
            return
        setattr(cfg, k, val)
        cfg.save()
        print(f"已保存：{k} = {val!r}")
        return
    print(f"配置文件：{CONFIG_PATH}")
    print(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="whispertype", description="气声输入：按住说，松开出字")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("devices", help="列出输入设备")
    d.set_defaults(func=cmd_devices)

    m = sub.add_parser("mic-check", help="录一段，检查电平和噪声")
    m.add_argument("--seconds", type=float, default=8.0)
    m.add_argument("--device")
    m.add_argument("--out", default="mic-check.wav")
    m.add_argument("--asr", action="store_true", help="顺便识别一次")
    m.add_argument("--model")
    m.set_defaults(func=cmd_mic_check)

    t = sub.add_parser("transcribe", help="识别一个 WAV，或现场录一段")
    t.add_argument("--wav")
    t.add_argument("--record", type=float, default=5.0)
    t.add_argument("--device")
    t.add_argument("--model")
    t.add_argument("--language")
    t.add_argument("--compute-device", choices=["auto", "cpu", "cuda"])
    t.add_argument("--compute-type")
    t.add_argument("--ref", help="参考文本，用来算 CER")
    t.set_defaults(func=cmd_transcribe, no_inject=True)

    doc = sub.add_parser("doctor", help="体检：依赖、设备、音量、热键、ASR")
    doc.add_argument("--record", type=float, default=1.0, help="采样秒数，0 跳过")
    doc.add_argument("--asr", action="store_true", help="用内置音频跑一次识别")
    doc.set_defaults(func=cmd_doctor)

    kt = sub.add_parser("keytest", help="按键体检：看 F8 实际被识别成什么")
    kt.set_defaults(func=cmd_keytest)

    r = sub.add_parser("run", help="常驻：热键说话 -> 识别 -> 输入")
    r.add_argument("--device")
    r.add_argument("--model")
    r.add_argument("--language")
    r.add_argument("--hotkey")
    r.add_argument("--compute-device", choices=["auto", "cpu", "cuda"])
    r.add_argument("--compute-type")
    r.add_argument("--toggle", action="store_true", help="按一下开始/再按结束")
    r.add_argument("--no-inject", action="store_true", help="只识别不输入")
    r.add_argument("--no-hud", action="store_true", help="不开小窗口，只在控制台输出")
    r.add_argument("--llm", action="store_true", help="启用 LLM 后处理（需 API key）")
    r.set_defaults(func=cmd_run, hud=None)

    c = sub.add_parser("config", help="查看/修改配置")
    c.add_argument("action", nargs="?", default="show", choices=["show", "set"])
    c.add_argument("pair", nargs="*")
    c.set_defaults(func=cmd_config)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        raise SystemExit(1)