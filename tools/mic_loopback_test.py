"""用扬声器播放已知句子，同时用麦克风录回来，验证"麦克风->识别"整条链路。

用法：
    python tools/mic_loopback_test.py --device BY-V20 --wav tests/data/tts_zh.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from whispertype import audio, vad
from whispertype.asr import ASR, make_asr
from whispertype.config import Config
from whispertype.metrics import report
from whispertype.textproc import clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--wav", default="tests/data/tts_zh.wav")
    ap.add_argument("--ref", default="今天天气不错，我们下午三点开会，请把电脑带上。")
    ap.add_argument("--model", default="small")
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--unmute", action="store_true", help="测试时临时取消扬声器静音（结束恢复）")
    args = ap.parse_args()

    cfg = Config.load().merged(model=args.model)
    device = audio.resolve_device(args.device) if args.device else None
    rec = audio.Recorder(device, 48000)
    print(f"录音 {args.seconds}s，同时播放 {args.wav} ……", flush=True)

    speaker = None
    old_mute = None
    if args.unmute:
        try:
            from pycaw.pycaw import AudioUtilities

            speaker = AudioUtilities.GetSpeakers().EndpointVolume
            old_mute = speaker.GetMute()
            speaker.SetMute(0, None)
        except Exception as e:
            print("取消静音失败：", e)

    rec.start()
    time.sleep(1.0)
    try:
        import winsound

        winsound.PlaySound(str(Path(args.wav)), winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print("播放失败：", e)
    time.sleep(max(0.0, args.seconds - 1.0))
    r = rec.stop()
    if speaker is not None and old_mute is not None:
        try:
            speaker.SetMute(old_mute, None)
        except Exception:
            pass

    print(f"录到 {r.seconds:.2f}s @ {r.samplerate} Hz")
    print(f"噪声底 {vad.noise_floor_db(r.samples, r.samplerate):.1f} dBFS，"
          f"峰值 {vad.peak_db(r.samples):.1f} dBFS，"
          f"RMS {vad.rms_db(r.samples):.1f} dBFS")
    audio.save_wav("tests/data/loopback-raw.wav", r.samples, r.samplerate)
    x = audio.prepare(r.samples, r.samplerate, cfg)
    audio.save_wav("tests/data/loopback-16k.wav", x, audio.TARGET_SR)

    asr = make_asr(cfg)
    res = asr.transcribe(x)
    text = clean(res.text)
    print(f"识别：{text}")
    print(f"耗时 {res.elapsed:.2f}s / 音频 {res.duration:.2f}s (RTF {res.rtf:.2f})")
    if args.ref:
        rep = report(args.ref, text)
        print(f"CER {rep['cer']:.1%}（编辑 {rep['edits']} / {rep['ref_chars']} 字）")


if __name__ == "__main__":
    main()