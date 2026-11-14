param(
    [ValidateRange(1, 900)][int]$DurationSeconds = 900,
    [switch]$VerifyOnly,
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'experiments\torque200\timed-train-wsl.sh'
$linuxLauncher = & wsl.exe -d Ubuntu --exec wslpath -a $launcher
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve WSL timed-training launcher' }
$nativeArgs = @('-d', 'Ubuntu', '--exec', '/bin/bash', $linuxLauncher.Trim(), '--duration', $DurationSeconds.ToString())
if ($VerifyOnly) { $nativeArgs += '--verify-only' }
if ($SelfTest) { $nativeArgs += '--self-test' }
& wsl.exe @nativeArgs
if ($LASTEXITCODE -ne 0) { throw 'Timed training failed; see saved status and output above' }
