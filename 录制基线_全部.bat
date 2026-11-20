@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
rem 录音设备名/编号可以跟在命令后面，例如：录制基线_全部.bat BY-V20
python tools\record_bench.py %*
echo.
echo Benchmark...
python tools\benchmark.py
pause