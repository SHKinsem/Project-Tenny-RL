$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'experiments\torque200\throughput-wsl.sh'
$linuxLauncher = & wsl.exe -d Ubuntu --exec wslpath -a $launcher
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve WSL throughput launcher' }
$nativeArgs = @('-d', 'Ubuntu', '--exec', '/bin/bash', $linuxLauncher.Trim())
& wsl.exe @nativeArgs
if ($LASTEXITCODE -ne 0) { throw 'Throughput probe failed; see output above' }
