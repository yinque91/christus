# Install Christus Autonomy at user login (observe mode by default).
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\install_startup.ps1

param([ValidateSet("observe", "control")][string]$Mode = "observe")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pythonw = (Get-Command pythonw.exe -ErrorAction Stop).Source
$entry = Join-Path $root "scripts\start_autonomy.py"
$run = '"{0}" "{1}" --mode {2}' -f $pythonw, $entry, $Mode

$key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
New-Item -Path $key -Force | Out-Null
Set-ItemProperty -Path $key -Name "ChristusAutonomy" -Value $run
Write-Host "Installed HKCU Run entry:"
Write-Host "  $run"
Write-Host "Emergency stop: Ctrl+Alt+Shift+Q"
Write-Host "Remove with: Remove-ItemProperty -Path '$key' -Name ChristusAutonomy"
