param(
    [ValidateRange(1, 1024)][int]$BatchSize = 2,
    [ValidateRange(1, 10000000)][int]$Steps = 16,
    [ValidateRange(1, 128)][int]$EvalEnvs = 2
)
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'experiments\torque200\run-wsl.sh'
$linuxLauncher = & wsl.exe -d Ubuntu --exec wslpath -a $launcher
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) { throw 'Cannot resolve torque200 WSL launcher path' }
$nativeArgs = @('-d', 'Ubuntu', '--exec', '/bin/bash', $linuxLauncher.Trim(),
    '--batch-size', $BatchSize, '--steps', $Steps, '--eval-envs', $EvalEnvs)
& wsl.exe @nativeArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) { throw 'Torque200 verification failed; inspect experiments/torque200/results' }
