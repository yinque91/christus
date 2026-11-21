@echo off
setlocal
set "ROOT=%~dp0"
set "PYW=C:\Users\13394\AppData\Local\Programs\Python\Python312\pythonw.exe"
set "PYTHONPATH=C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR\.venv\Lib\site-packages;%ROOT%;%ROOT%\CHRISTUS-DATA-01"
cd /d "%ROOT%"
start "" "%PYW%" christus_mirror\lab_app.py
