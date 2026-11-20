"""用百炼的语音模型合成「气声」测试音频（ASR 评测 / 数据增强用）。

链路：
    文本 → 百炼 TTS（qwen-audio-3.0-tts-plus，带「耳语」指令）
         → 气声化：倒谱平滑保留频谱包络，把谐波激励换成噪声（去掉基频）
         → 24kHz 单声道 WAV

为什么还要自己加一步气声化：API 的 TTS 靠指令只能做到「轻声」——实测 HNR 从 6.4dB 降到
2.5dB、有声帧从 25% 降到 12%，但仍有基频；真实气声是 0 有声帧、HNR −5dB 左右。
气声化之后 HNR −4～−7dB、有声帧 0%，与真实气声的声学特征对齐（见 docs/合成气声_评测.md）。

用法：
    python tools/synth_whisper.py                       # data/bench_zh.tsv 全部 50 句
    python tools/synth_whisper.py --limit 10            # 先跑 10 句试试
    python tools/synth_whisper.py --mode normal         # 对照：正常朗读（不气声化）
    python tools/synth_whisper.py --out bench-synth --no-whisperize

跑完评测：
    python tools/benchmark.py --dir bench-synth
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import websocket

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "bench_zh.tsv"

API_HOST = "token-plan.cn-beijing.maas.aliyuncs.com"
WS_URL = f"wss://{API_HOST}/api-ws/v1/inference"
MODEL = "qwen-audio-3.0-tts-plus"
VOICE = "longanqian"
SR = 24000

# 指令：越具体越接近气声（实测 HNR 从 6.4 → 2.5dB）
WHISPER_INSTRUCTION = "耳语，只用气流，声带不要振动"


def load_key() -> str:
    key = os.environ.get("ALI_BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if key:
        return key
    cred = Path.home() / ".dsh" / ".credentials.yaml"
    if cred.exists():
        import yaml
        data = yaml.safe_load(cred.read_text(encoding="utf-8"))
        key = (data.get("refs") or {}).get("ALI_BAILIAN_API_KEY")
        if key:
            return key
    raise SystemExit("找不到百炼 API key：设 ALI_BAILIAN_API_KEY，或让 DSH 配好 .credentials.yaml")


class TTS:
    def __init__(self, key: str) -> None:
        self.key = key
        self.calls = 0
        self.chars = 0

    def say(self, text: str, instruction: str | None = None, timeout: int = 90) -> np.ndarray:
        """合成一句话，返回 24k 单声道 float32。"""
        task_id = uuid.uuid4().hex
        params = {"voice": VOICE, "format": "wav", "sample_rate": SR}
        if instruction:
            params["instruction"] = instruction
        ws = websocket.create_connection(
            WS_URL, timeout=20, header=[f"Authorization: Bearer {self.key}"],
            sslopt={"cert_reqs": ssl.CERT_REQUIRED}, http_proxy_host=None)
        ws.settimeout(30)
        try:
            ws.send(json.dumps({"header": {"action": "run-task", "task_id": task_id,
                                           "streaming": "duplex"},
                                "payload": {"model": MODEL, "task_group": "audio", "task": "tts",
                                            "function": "SpeechSynthesizer",
                                            "input": {"text": text}, "parameters": params}}))
            audio = bytearray()
            failed = ""
            deadline = time.time() + timeout
            while time.time() < deadline:
                raw = ws.recv()
                if isinstance(raw, bytes):
                    audio += raw
                    continue
                ev = json.loads(raw)
                header = ev.get("header") or {}
                event = header.get("event")
                if event == "task-failed":
                    failed = json.dumps(ev.get("payload") or {}, ensure_ascii=False)[:300]
                    break
                if event == "task-finished":
                    break
                if event == "task-started":
                    ws.send(json.dumps({"header": {"action": "finish-task", "task_id": task_id,
                                                   "streaming": "duplex"},
                                        "payload": {"input": {}}}))
            if failed:
                raise RuntimeError(f"TTS 失败：{failed}")
            if not audio:
                raise RuntimeError("TTS 没有返回音频")
            self.calls += 1
            self.chars += len(text)
            return pcm_from_wav(bytes(audio)) if bytes(audio[:4]) == b"RIFF" else \
                np.frombuffer(bytes(audio), dtype=np.int16).astype(np.float32) / 32768.0
        finally:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass


def pcm_from_wav(blob: bytes) -> np.ndarray:
    import io
    with wave.open(io.BytesIO(blob), "rb") as w:
        n, ch = w.getnframes(), w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, ch).mean(axis=1) if ch > 1 else data


def whisperize(x: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256,
               lifter_ms: float = 2.2, seed: int = 0) -> np.ndarray:
    """倒谱平滑出频谱包络，用噪声激励重合成（去掉基频与谐波结构）。"""
    rng = np.random.default_rng(seed)
    win = np.hanning(n_fft)
    pad = n_fft
    xp = np.pad(x.astype(np.float64), (pad, pad))
    n_frames = 1 + (len(xp) - n_fft) // hop
    out = np.zeros_like(xp)
    wsum = np.zeros_like(xp)
    q = max(4, int(lifter_ms * sr / 1000))
    for i in range(n_frames):
        frame = xp[i * hop:i * hop + n_fft] * win
        mag = np.abs(np.fft.rfft(frame)) + 1e-8
        cep = np.fft.irfft(np.log(mag))
        cep_f = cep.copy()
        cep_f[q:len(cep_f) - q + 1] = 0.0            # 只留低倒频 = 频谱包络
        env = np.exp(np.fft.rfft(cep_f).real)
        phase = rng.uniform(-np.pi, np.pi, len(env))  # 随机相位 = 噪声激励
        out[i * hop:i * hop + n_fft] += np.fft.irfft(env * np.exp(1j * phase)) * win
        wsum[i * hop:i * hop + n_fft] += win ** 2
    out = (out / np.maximum(wsum, 1e-8))[pad:len(out) - pad]
    return out.astype(np.float32)


def acoustic(x: np.ndarray, sr: int) -> dict:
    """气声的声学体检：HNR（谐噪比）、有声帧比例。真实气声应 voiced≈0、HNR≈−5dB。"""
    win, hop = int(sr * 0.032), int(sr * 0.010)
    hnr, energy = [], []
    for i in range(0, len(x) - win, hop):
        f = x[i:i + win]
        energy.append(float(np.mean(f ** 2)))
        if energy[-1] < 1e-8:
            continue
        fx = f - f.mean()
        ac = np.correlate(fx, fx, "full")[len(fx) - 1:]
        ac /= (ac[0] + 1e-12)
        lo, hi = int(sr / 400), int(sr / 70)
        peak = float(np.max(ac[lo:hi])) if hi < len(ac) else 0.0
        hnr.append(10 * np.log10(max(peak, 1e-6) / max(1e-6, 1 - peak)))
    h = np.array(hnr)
    e = np.array(energy)
    active = e > (np.max(e) * 0.01) if len(e) else np.zeros(0, bool)
    voiced = active[:len(h)] & (h > 8)
    return {
        "hnr_db": round(float(np.median(h)), 1) if len(h) else None,
        "voiced_ratio": round(float(voiced.mean()), 3) if len(voiced) else 0.0,
        "rms_dbfs": round(float(10 * np.log10(np.mean(x ** 2) + 1e-12)), 1),
        "duration": round(len(x) / sr, 2),
    }


def save_wav(path: Path, x: np.ndarray, sr: int) -> None:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def load_manifest(limit: int = 0) -> list[tuple[str, str, str]]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, cat, text = line.split("\t", 2)
            rows.append((sid, cat, text))
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser(description="合成气声测试音频（百炼 TTS + 气声化）")
    ap.add_argument("--out", default="bench-synth", help="输出目录，默认 bench-synth/")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 句")
    ap.add_argument("--mode", default="whisper", choices=["whisper", "soft", "normal"],
                    help="whisper=气声化(默认) / soft=只要 TTS 轻声 / normal=正常朗读")
    ap.add_argument("--no-whisperize", action="store_true", help="不做气声化（等于 --mode soft）")
    ap.add_argument("--lifter", type=float, default=2.2, help="倒谱平滑点数，越小越闷")
    ap.add_argument("--force", action="store_true", help="已有文件也重做")
    args = ap.parse_args()

    mode = "soft" if args.no_whisperize else args.mode
    out_dir = ROOT / args.out
    tts = TTS(load_key())
    rows = load_manifest(args.limit)
    print(f"合成 {len(rows)} 句 · 模式 {mode} · 输出 {out_dir}")

    meta = []
    t0 = time.time()
    for sid, cat, text in rows:
        wav = out_dir / f"{sid}.wav"
        if wav.exists() and not args.force:
            print(f"  {sid} 已存在，跳过")
            continue
        try:
            if mode == "normal":
                audio = tts.say(text, instruction=None)
            else:
                audio = tts.say(text, instruction=WHISPER_INSTRUCTION)
            if mode == "whisper":
                audio = whisperize(audio, SR, lifter_ms=args.lifter, seed=int(sid) if sid.isdigit() else 0)
                audio = audio * 0.6                 # 气声本来就比正常说话低 10~20dB
            save_wav(wav, audio, SR)
            m = acoustic(audio, SR)
            meta.append({"id": sid, "category": cat, "ref": text, "mode": mode,
                         "instruction": WHISPER_INSTRUCTION if mode != "normal" else "",
                         "lifter_ms": args.lifter if mode == "whisper" else None, **m})
            flag = "" if mode == "normal" else ("  ✓" if (m["voiced_ratio"] or 0) < 0.05 else "  ⚠ 仍有基频")
            print(f"  {sid} {m['duration']:5.2f}s HNR {m['hnr_db']}dB voiced {m['voiced_ratio']:.0%}{flag}")
        except Exception as e:  # noqa: BLE001
            print(f"  {sid} 失败：{type(e).__name__}: {e}")

    if meta:
        meta_path = out_dir / "synth_meta.json"
        meta_path.write_text(json.dumps({"mode": mode, "model": MODEL, "voice": VOICE,
                                         "items": meta}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        voiced = [m["voiced_ratio"] for m in meta if m.get("voiced_ratio") is not None]
        hnrs = [m["hnr_db"] for m in meta if m.get("hnr_db") is not None]
        print(f"\n{len(meta)} 句 · 用时 {time.time() - t0:.0f}s · "
              f"TTS {tts.calls} 次 / {tts.chars} 字")
        if mode != "normal":
            print(f"有声帧比例 中位 {np.median(voiced):.1%}（真实气声 0%）· "
                  f"HNR 中位 {np.median(hnrs):.1f}dB（真实气声 −5dB 左右）")
        print(f"元数据: {meta_path}")
        print(f"下一步评测: python tools/benchmark.py --dir {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())