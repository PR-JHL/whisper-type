"""气声对照测试菜单：录音 + 评测，避免 cmd 批处理解析中文的问题。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IDS = "012,021,013,011,004,003,010,016,031,044"


def run(args: list[str]) -> int:
    return subprocess.call([sys.executable, *args], cwd=str(ROOT))


def main() -> None:
    print("=" * 46)
    print("  气声基线对照测试（同一批 10 句）")
    print("=" * 46)
    print("  1  正常音量（对照组）")
    print("  2  贴近麦克风 3-5cm 的气声")
    print("  3  轻声但出声（有声带振动，不是纯气声）")
    print("  4  只跑评测，不录音")
    print("  0  退出")
    print()
    try:
        choice = input("选择 [0-4]: ").strip()
    except EOFError:
        return

    if choice == "0" or not choice:
        return
    mapping = {"1": "bench-normal", "2": "bench-close", "3": "bench-voiced", "4": "bench-normal"}
    if choice not in mapping:
        print("没听懂选项。")
        return
    out = mapping[choice]

    if choice != "4":
        code = run(
            [
                "tools/record_bench.py",
                "--device",
                "BY-V20",
                "--ids",
                IDS,
                "--out",
                out,
            ]
        )
        if code != 0:
            print("\n录音脚本出错（或麦克风没信号），把上面的报错截图发我。")
            return
        if not list((ROOT / out).glob("*.wav")):
            print("\n没有录到 WAV，停止。")
            return

    print("\n跑评测...")
    run(["tools/benchmark.py", "--dir", out])
    print(f"\n报告：{(ROOT / 'docs' / f'bench_report_{out}.md')}")


if __name__ == "__main__":
    main()