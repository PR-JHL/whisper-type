"""把识别结果送进当前前台窗口（Windows）。

- 短文本：SendInput + KEYEVENTF_UNICODE，逐字输入，不经过输入法
- 长文本：写剪贴板 + Ctrl+V，快很多，然后恢复剪贴板
"""

from __future__ import annotations

import sys
import time

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class _UNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _UNION)]

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    _user32.SendInput.restype = wintypes.UINT

    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    INPUT_KEYBOARD = 1

    def _key(vk: int = 0, scan: int = 0, flags: int = 0) -> "INPUT":
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.ki = KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)
        return inp

    def _send(*inputs: "INPUT") -> int:
        arr = (INPUT * len(inputs))(*inputs)
        return _user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))

    def type_unicode(text: str, interval: float = 0.002) -> None:
        for ch in text:
            code = ord(ch)
            units = (code,)
            if code > 0xFFFF:
                code -= 0x10000
                units = (0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF))
            for u in units:
                _send(_key(scan=u, flags=KEYEVENTF_UNICODE))
                _send(_key(scan=u, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
            if interval:
                time.sleep(interval)

    def _tap(vk: int, modifiers: tuple[int, ...] = ()) -> None:
        for m in modifiers:
            _send(_key(vk=m))
        _send(_key(vk=vk))
        _send(_key(vk=vk, flags=KEYEVENTF_KEYUP))
        for m in reversed(modifiers):
            _send(_key(vk=m, flags=KEYEVENTF_KEYUP))

    def paste_text(text: str, restore_delay: float = 0.25) -> None:
        import pyperclip

        old = ""
        try:
            old = pyperclip.paste()
        except Exception:
            pass
        pyperclip.copy(text)
        time.sleep(0.03)
        _tap(ord("V"), modifiers=(0x11,))  # Ctrl+V
        time.sleep(restore_delay)
        try:
            pyperclip.copy(old)
        except Exception:
            pass

else:  # pragma: no cover - 非 Windows 只做空实现，方便跑测试

    def type_unicode(text: str, interval: float = 0.002) -> None:
        print(text)

    def paste_text(text: str, restore_delay: float = 0.25) -> None:
        print(text)


def type_text(text: str, paste_threshold: int = 24) -> None:
    if not text:
        return
    if len(text) >= paste_threshold:
        try:
            paste_text(text)
            return
        except Exception:
            pass
    type_unicode(text)