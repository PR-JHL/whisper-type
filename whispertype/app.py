"""运行时：录音 -> 识别 -> 注入，外加一个小 HUD。"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from . import audio, llm, vad
from .audio import Recorder
from .config import Config
from .inject import type_text
from .textproc import clean, is_hallucination, prompt_echo


@dataclass
class State:
    kind: str  # loading | idle | recording | busy | done | error
    text: str = ""
    detail: str = ""


class Controller:
    def __init__(self, cfg: Config, asr, on_state) -> None:
        self.cfg = cfg
        self.asr = asr
        self.on_state = on_state
        self.rec = Recorder(cfg.device, cfg.samplerate)
        self._lock = threading.Lock()
        self._recording = False
        self._busy = False
        self.ready = False
        self._last_text = ""

    # ---------------------------------------------------------------- 模型
    def load_model_async(self) -> None:
        self.on_state(State("loading", detail=f"模型 {self.cfg.model}，首次会下载"))
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self) -> None:
        try:
            self.asr.load()
            self.ready = True
            self.on_state(State("idle", detail=self.hint()))
        except Exception as e:  # 模型/依赖问题
            self.on_state(State("error", detail=f"模型加载失败：{e}"))

    def hint(self) -> str:
        key = self.cfg.hotkey.upper()
        if self.cfg.mode == "toggle":
            return f"按 {key} 开始 / 再按结束"
        return f"按住 {key} 说话，松开识别"

    # ---------------------------------------------------------------- 录音
    def press(self) -> None:
        if not self.ready:
            self.on_state(State("loading", detail="模型还在加载…"))
            return
        with self._lock:
            if self._recording or self._busy:
                return
            try:
                self.rec.start()
            except Exception as e:
                self.on_state(State("error", detail=f"打不开麦克风：{e}"))
                return
            self._recording = True
        self.on_state(State("recording", detail="录音中…松开结束"))

    def release(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            rec = self.rec.stop()
            self._busy = True
        if rec.seconds < self.cfg.min_utterance:
            self._busy = False
            self.on_state(State("idle", detail=f"太短（{rec.seconds:.2f}s），忽略"))
            return
        self.on_state(State("busy", detail="识别中…"))
        threading.Thread(target=self._work, args=(rec,), daemon=True).start()

    def _work(self, rec) -> None:
        try:
            x = audio.prepare(rec.samples, rec.samplerate, self.cfg)
            ratio = vad.speech_ratio(x, audio.TARGET_SR, self.cfg.vad_margin_db)
            if len(x) < int(0.2 * audio.TARGET_SR) or ratio < 0.06 or not vad.has_speech(
                x, audio.TARGET_SR, self.cfg.vad_margin_db
            ):
                self.on_state(State("idle", detail="没听到说话"))
                return
            res = self.asr.transcribe(x)
            text = clean(res.text)
            if is_hallucination(text, res.duration) or prompt_echo(text, self.cfg.initial_prompt):
                text = ""
            elif text and self.cfg.llm_enabled and llm.available(self.cfg):
                text = llm.correct(text, self.cfg)
            if text and self.cfg.inject:
                type_text(text, self.cfg.paste_threshold)
            self._last_text = text
            detail = (
                f"音频 {res.duration:.1f}s · 识别 {res.elapsed:.2f}s "
                f"(RTF {res.rtf:.2f})"
            )
            self.on_state(State("done" if text else "idle", text=text, detail=detail))
        except Exception as e:
            self.on_state(State("error", detail=f"识别失败：{e}"))
        finally:
            with self._lock:
                self._busy = False

    @property
    def last_text(self) -> str:
        return self._last_text


# ---------------------------------------------------------------------- HUD
BG = "#0f1319"
PANEL = "#171d26"
TEXT = "#e9eef7"
MUTED = "#8b98aa"
GREEN = "#3ddc97"
ACCENT_DIM = "#16382c"
AMBER = "#ffb454"
RED = "#ff6b6b"
BLUE = "#5aa9ff"


class HUD:
    def __init__(self, cfg: Config) -> None:
        import tkinter as tk

        self.tk = tk
        self.cfg = cfg
        self.q: queue.Queue[State] = queue.Queue()
        self.controller: Controller | None = None
        self.root = tk.Tk()
        self.root.title("气声输入 · whisper type")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        w, h = 460, 208
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{max(0, sw - w - 40)}+{max(0, sh - h - 80)}")

        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=16, pady=(14, 4))
        self.dot = tk.Label(header, text="●", bg=BG, fg=MUTED, font=("Segoe UI", 16))
        self.dot.pack(side="left")
        self.title = tk.Label(
            header, text="待机", bg=BG, fg=TEXT, font=("Microsoft YaHei UI", 13, "bold")
        )
        self.title.pack(side="left", padx=(6, 0))
        self.detail = tk.Label(
            header, text="", bg=BG, fg=MUTED, font=("Microsoft YaHei UI", 9)
        )
        self.detail.pack(side="right")

        self.text = tk.Label(
            self.root,
            text="",
            bg=PANEL,
            fg=TEXT,
            font=("Microsoft YaHei UI", 12),
            wraplength=420,
            justify="left",
            anchor="nw",
            padx=12,
            pady=10,
        )
        self.text.pack(fill="both", expand=True, padx=16, pady=8)

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=16, pady=(0, 12))
        self.hint = tk.Label(
            footer,
            text=f"{cfg.hotkey.upper()} · {cfg.mode} · 模型 {cfg.model}",
            bg=BG,
            fg=MUTED,
            font=("Microsoft YaHei UI", 9),
        )
        self.hint.pack(side="left")
        tk.Button(
            footer,
            text="退出",
            command=self.root.destroy,
            bg=PANEL,
            fg=TEXT,
            relief="flat",
            padx=10,
        ).pack(side="right")
        self.talk = tk.Button(
            footer,
            text="按住说话",
            bg=ACCENT_DIM,
            fg=GREEN,
            relief="flat",
            padx=10,
        )
        self.talk.pack(side="right", padx=(0, 8))
        self.talk.bind("<ButtonPress-1>", self._talk_down)
        self.talk.bind("<ButtonRelease-1>", self._talk_up)

        self.root.after(60, self._poll)

    def attach(self, controller: "Controller") -> None:
        self.controller = controller

    def _talk_down(self, _event) -> None:
        if self.controller is not None:
            self.controller.press()

    def _talk_up(self, _event) -> None:
        if self.controller is not None:
            self.controller.release()

    def set_state(self, st: State) -> None:
        self.q.put(st)

    def _poll(self) -> None:
        try:
            while True:
                st = self.q.get_nowait()
                self._apply(st)
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    def _apply(self, st: State) -> None:
        color = {
            "loading": AMBER,
            "idle": MUTED,
            "recording": RED,
            "busy": BLUE,
            "done": GREEN,
            "error": RED,
        }.get(st.kind, MUTED)
        label = {
            "loading": "加载中",
            "idle": "待机",
            "recording": "录音",
            "busy": "识别中",
            "done": "已输入",
            "error": "出错",
        }.get(st.kind, st.kind)
        self.dot.configure(fg=color)
        self.title.configure(text=label)
        self.detail.configure(text=st.detail or "", fg=color if st.kind in ("error",) else MUTED)
        if st.text:
            self.text.configure(text=st.text)
        elif st.kind in ("recording", "loading"):
            self.text.configure(text="")
        if st.kind == "recording":
            self.text.configure(text="…", fg=MUTED)
        elif st.kind == "done":
            self.text.configure(fg=TEXT)

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------- 运行
def _log_state(st: State) -> None:
    try:
        from .config import APP_DIR

        APP_DIR.mkdir(parents=True, exist_ok=True)
        with open(APP_DIR / "run.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{st.kind}] {st.text} {st.detail}\n")
    except Exception:
        pass


def _trim_log() -> None:
    try:
        from .config import APP_DIR

        p = APP_DIR / "run.log"
        if p.exists() and p.stat().st_size > 2_000_000:
            p.write_text("", encoding="utf-8")
    except Exception:
        pass


def run(cfg: Config, asr, hud: bool | None = None) -> None:
    use_hud = cfg.hud if hud is None else hud
    _trim_log()
    if use_hud:
        ui = HUD(cfg)

        def on_state(st: State) -> None:
            _log_state(st)
            ui.set_state(st)

        controller = Controller(cfg, asr, on_state)
        ui.attach(controller)
        hk = _make_hotkey(cfg, controller)
        hk.start()
        controller.load_model_async()
        try:
            ui.run()
        finally:
            hk.stop()
            try:
                controller.rec.stop()
            except Exception:
                pass
        return

    def on_state(st: State) -> None:
        _log_state(st)
        line = f"[{st.kind}] {st.text or ''} {st.detail or ''}".rstrip()
        print(line, flush=True)

    controller = Controller(cfg, asr, on_state)
    hk = _make_hotkey(cfg, controller)
    hk.start()
    controller.load_model_async()
    print(f"监听 {cfg.hotkey.upper()}（{cfg.mode}）。Ctrl+C 退出。", flush=True)
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        hk.stop()
        try:
            controller.rec.stop()
        except Exception:
            pass


def _make_hotkey(cfg: Config, controller: Controller):
    from .hotkey import Hotkey

    try:
        return Hotkey(cfg.hotkey, controller.press, controller.release, cfg.mode)
    except Exception as e:
        raise SystemExit(f"热键不可用：{e}")