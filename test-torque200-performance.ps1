param([string]$WebUrl = 'http://127.0.0.1:8770')
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'experiments\torque200\performance-wsl.sh'
$linuxLauncher = & wsl.exe -d Ubuntu --exec wslpath -a $launcher
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve WSL performance launcher' }
$nativeArgs = @('-d', 'Ubuntu', '--exec', '/bin/bash', $linuxLauncher.Trim(), '--web-url', $WebUrl)
& wsl.exe @nativeArgs
if ($LASTEXITCODE -ne 0) { throw 'Torque200 performance comparison failed; see output above' }
