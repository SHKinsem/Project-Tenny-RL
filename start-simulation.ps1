$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'tasks\start-sim-wsl.sh'
$linuxLauncher = & wsl.exe -d Ubuntu --exec wslpath -a $launcher
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve WSL launcher path' }
& wsl.exe -d Ubuntu --exec /bin/bash $linuxLauncher.Trim()
if ($LASTEXITCODE -ne 0) { throw 'Simulation exited with an error; see output above' }
