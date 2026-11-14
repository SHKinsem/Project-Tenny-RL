param(
    [ValidateRange(1, 1024)][int]$BatchSize = 2,
    [ValidateRange(1, 10000000)][int]$Steps = 16,
    [ValidateRange(1, 128)][int]$EvalEnvs = 2
)
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'tasks\run-gpu-wsl.sh'
$linuxLauncher = & wsl.exe -d Ubuntu --exec wslpath -a $launcher
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve WSL launcher path' }
$nativeArgs = @('-d', 'Ubuntu', '--exec', '/bin/bash', $linuxLauncher.Trim(),
    '--batch-size', $BatchSize, '--steps', $Steps, '--eval-envs', $EvalEnvs)
& wsl.exe @nativeArgs
if ($LASTEXITCODE -ne 0) { throw 'GPU test failed; see output and WSL logs' }
