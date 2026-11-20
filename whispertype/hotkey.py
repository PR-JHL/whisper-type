"""全局热键：按住说话（hold）或按一下切换（toggle）。

支持键盘（pynput.keyboard）和鼠标侧键（pynput.mouse）。
Windows 下普通桌面会话即可，不需要管理员权限。
"""

from __future__ import annotations

from typing import Callable


def _keyboard():
    from pynput import keyboard

    return keyboard


def _mouse():
    from pynput import mouse

    return mouse


def parse_key(name: str):
    kb = _keyboard()
    n = (name or "").strip().lower()
    aliases: dict[str, object] = {
        "f1": kb.Key.f1, "f2": kb.Key.f2, "f3": kb.Key.f3, "f4": kb.Key.f4,
        "f5": kb.Key.f5, "f6": kb.Key.f6, "f7": kb.Key.f7, "f8": kb.Key.f8,
        "f9": kb.Key.f9, "f10": kb.Key.f10, "f11": kb.Key.f11, "f12": kb.Key.f12,
        "caps_lock": kb.Key.caps_lock,
        "ctrl_r": kb.Key.ctrl_r,
        "ctrl_l": kb.Key.ctrl_l,
        "alt_r": kb.Key.alt_r,
        "alt_l": kb.Key.alt_l,
        "shift_r": kb.Key.shift_r,
        "shift_l": kb.Key.shift_l,
        "space": kb.Key.space,
        "tab": kb.Key.tab,
        "pause": kb.Key.pause,
        "scroll_lock": kb.Key.scroll_lock,
        "insert": kb.Key.insert,
        "home": kb.Key.home,
        "end": kb.Key.end,
        "page_up": kb.Key.page_up,
        "page_down": kb.Key.page_down,
    }
    if n in aliases:
        return aliases[n]
    mouse_aliases = {
        "mouse4": _mouse().Button.x1,
        "mouse5": _mouse().Button.x2,
        "middle": _mouse().Button.middle,
        "mouse_middle": _mouse().Button.middle,
    }
    if n in mouse_aliases:
        return mouse_aliases[n]
    if len(n) == 1:
        return kb.KeyCode.from_char(n)
    raise ValueError(
        f"不认识的热键 {name!r}；可用 f1–f12、caps_lock、ctrl_r、alt_r、shift_r、"
        f"mouse4、mouse5、middle 或单个字母"
    )


class Hotkey:
    def __init__(
        self,
        name: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        mode: str = "hold",
    ) -> None:
        self.name = name
        self.target = parse_key(name)
        self.on_start = on_start
        self.on_stop = on_stop
        self.mode = mode if mode in ("hold", "toggle") else "hold"
        self._down = False
        self._toggled = False
        self._listener = None
        self._is_mouse = name.lower().startswith(("mouse", "middle"))

    def start(self) -> None:
        if self._is_mouse:
            self._listener = _mouse().Listener(on_click=self._click)
        else:
            self._listener = _keyboard().Listener(
                on_press=self._press, on_release=self._release
            )
        self._listener.daemon = True
        self._listener.start()

    def _fire_start(self) -> None:
        if self.mode == "toggle":
            if self._toggled:
                self._toggled = False
                self.on_stop()
            else:
                self._toggled = True
                self.on_start()
        else:
            if not self._down:
                self._down = True
                self.on_start()

    def _fire_stop(self) -> None:
        if self.mode == "hold" and self._down:
            self._down = False
            self.on_stop()

    def _press(self, key) -> None:
        try:
            if key != self.target:
                return
        except Exception:
            return
        self._fire_start()

    def _release(self, key) -> None:
        try:
            if key != self.target:
                return
        except Exception:
            return
        self._fire_stop()

    def _click(self, x, y, button, pressed) -> None:
        if button != self.target:
            return
        if pressed:
            self._fire_start()
        else:
            self._fire_stop()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


def suggest(key) -> str:
    """把 pynput 的键对象映射成配置里能写的名字。"""
    name = getattr(key, "name", None)
    if name:
        return name
    char = getattr(key, "char", None)
    if char:
        return char
    return repr(key)