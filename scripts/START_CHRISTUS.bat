@echo off
setlocal
set "ROOT=%~dp0"
set "PY=C:\Users\13394\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" (
  echo Python not found: %PY%
  pause
  exit /b 1
)
cd /d "%ROOT%"
"%PY%" christus_desktop\launcher.py start
pause
