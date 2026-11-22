@echo off
setlocal
set "ROOT=%~dp0.."
cd /d "%ROOT%"
python scripts\start_autonomy.py --mode observe
pause
