param(
    [ValidateRange(1024, 65535)][int]$Port = 8770,
    [ValidateSet('original', 'torque200')][string]$Mode = 'torque200'
)
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'experiments\torque200\web-preview-wsl.sh'
$linuxLauncher = & wsl.exe -d Ubuntu --exec wslpath -a $launcher
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve WSL web launcher' }
$wslAddresses = & wsl.exe -d Ubuntu --exec hostname -I
if ($LASTEXITCODE -ne 0) { throw 'Cannot read WSL address' }
$webAddress = ($wslAddresses.Trim() -split '\s+')[0]
Write-Host "Browser URL after warm-up: http://${webAddress}:$Port/"
$nativeArgs = @('-d', 'Ubuntu', '--exec', '/bin/bash', $linuxLauncher.Trim(), '--port', [string]$Port, '--mode', $Mode)
& wsl.exe @nativeArgs
if ($LASTEXITCODE -ne 0) { throw 'Torque200 web preview failed; see output above' }
