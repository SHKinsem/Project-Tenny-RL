param([ValidateRange(0, 86400)][double]$Seconds = 0)
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'experiments\torque200\preview-wsl.sh'
$linuxLauncher = & wsl.exe -d Ubuntu --exec wslpath -a $launcher
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve WSL preview launcher' }
$nativeArgs = @('-d', 'Ubuntu', '--exec', '/bin/bash', $linuxLauncher.Trim(), '--seconds', $Seconds.ToString([cultureinfo]::InvariantCulture))
& wsl.exe @nativeArgs
if ($LASTEXITCODE -ne 0) { throw 'Torque200 preview failed; see output above' }
