"""本地 HTTP 识别服务，给 DSH 的 speech provider 调用。

    python -m whispertype.server --port 17321 --model small

接口（只绑定 127.0.0.1，音频不落盘）：
    GET  /health                    -> {"ok":true,"model":"small","uptime":...}
    POST /transcribe?language=zh    -> 请求体是 16k 单声道 PCM16 WAV，返回 {"text":...}

刻意只用标准库：不引入 FastAPI/Flask，DSH 那边拉起一个进程就能用。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import audio, vad
from .asr import ASR
from .config import Config
from .textproc import clean, is_hallucination, prompt_echo

STATE: dict = {"asr": None, "cfg": None, "started": time.time()}
LOCK = threading.Lock()
LOG_PATH = Path(os.environ.get("USERPROFILE") or Path.home()) / ".whispertype" / "server.log"


def log_line(message: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > 2_000_000:
            LOG_PATH.write_text("", encoding="utf-8")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except Exception:
        pass


class Handler(BaseHTTPRequestHandler):
    server_version = "whispertype/0.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[server] " + (fmt % args) + "\n")
        sys.stderr.flush()

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            cfg: Config = STATE["cfg"]
            self._json(
                200,
                {
                    "ok": True,
                    "model": cfg.model,
                    "compute": f"{cfg.compute_device}/{cfg.compute_type}",
                    "uptime": round(time.time() - STATE["started"], 1),
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/transcribe":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8 * 1024 * 1024:
                self._json(400, {"error": "bad content length"})
                return
            body = self.rfile.read(length)
            language = (parse_qs(parsed.query).get("language") or ["zh"])[0]
            cfg: Config = STATE["cfg"]
            asr: ASR = STATE["asr"]

            rec = audio.decode_wav_bytes(body)
            log_line(
                f"POST bytes={length} sr={rec.samplerate} ch=1 sec={rec.seconds:.2f} lang={language}"
            )
            x = audio.prepare(rec.samples, rec.samplerate, cfg)
            ratio = vad.speech_ratio(x, audio.TARGET_SR, cfg.vad_margin_db)
            if len(x) < int(0.2 * audio.TARGET_SR) or ratio < 0.06:
                log_line(f"  -> empty (prepared={len(x)/audio.TARGET_SR:.2f}s ratio={ratio:.2f})")
                self._json(
                    200,
                    {"text": "", "duration": round(len(x) / audio.TARGET_SR, 2), "reason": "no_speech"},
                )
                return

            with LOCK:  # ctranslate2 串行推理，避免多请求抢 CPU
                t0 = time.perf_counter()
                res = asr.transcribe(x)
                elapsed = time.perf_counter() - t0
            text = clean(res.text)
            if is_hallucination(text, res.duration) or prompt_echo(text, cfg.initial_prompt):
                text = ""
            log_line(
                f"  -> ok {res.duration:.2f}s / {elapsed:.2f}s text={json.dumps(text, ensure_ascii=False)}"
            )
            self._json(
                200,
                {
                    "text": text,
                    "duration": round(res.duration, 2),
                    "elapsed": round(elapsed, 2),
                    "rtf": round(elapsed / max(1e-6, res.duration), 2),
                    "language": language,
                },
            )
        except Exception as e:  # 任何解析/推理错误都返回 JSON，方便 DSH 报错
            log_line("  -> ERROR " + traceback.format_exc().replace("\n", " | "))
            sys.stderr.write("[server] error: " + traceback.format_exc() + "\n")
            sys.stderr.flush()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="whispertype 本地识别服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=17321)
    ap.add_argument("--model", default=None)
    ap.add_argument("--compute-device", default=None)
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--language", default=None)
    args = ap.parse_args(argv)

    cfg = Config.load().merged(
        model=args.model,
        compute_device=args.compute_device,
        compute_type=args.compute_type,
        language=args.language,
    )
    cfg.apply_env()
    asr = ASR(cfg)
    asr.load()
    STATE["cfg"] = cfg
    STATE["asr"] = asr
    STATE["started"] = time.time()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"[server] ready http://{args.host}:{args.port} model={cfg.model}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()