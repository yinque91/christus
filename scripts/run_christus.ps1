param(
    [Parameter(Mandatory=$true)][string]$Out,
    [int]$Steps = 1000,
    [ValidateSet('safe','integration')][string]$Profile = 'safe',
    [switch]$Resume,
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $runArgs = @('-m','christus','--out',$Out,'--steps',"$Steps",'--profile',$Profile)
    if ($Resume) { $runArgs += '--resume' }
    & $Python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "CHRISTUS exited with $LASTEXITCODE" }
} finally { Pop-Location }
