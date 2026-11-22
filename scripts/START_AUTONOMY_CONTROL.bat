@echo off
setlocal
set "ROOT=%~dp0.."
cd /d "%ROOT%"
echo Christus autonomy: control mode.
echo Actions are allowlist/safety-checked. Emergency stop: Ctrl+Alt+Shift+Q.
python scripts\start_autonomy.py --mode control
pause
