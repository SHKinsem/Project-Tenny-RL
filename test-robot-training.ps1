param(
    [ValidateSet('selected', 'original', 'torque200')][string]$Mode = 'selected',
    [ValidateRange(1, 1024)][int]$BatchSize = 2,
    [ValidateRange(1, 10000000)][int]$Steps = 16,
    [ValidateRange(1, 128)][int]$EvalEnvs = 2,
    [switch]$DescribeOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'tasks/resolve-robot-mode.ps1')
$resolved = Resolve-RobotMode -Mode $Mode
$entry = if ($resolved -eq 'original') { 'test-gpu-training.ps1' } else { 'test-torque200-training.ps1' }
$description = if ($resolved -eq 'original') { 'Original: leg position + wheel velocity; 100 Hz; 26 observations' } else { 'Torque experiment: four torques; 200 Hz; 24 observations' }
Write-Host $description
Write-Host "Bounded plumbing check: batch=$BatchSize, requested steps=$Steps, eval environments=$EvalEnvs"
Write-Host "Entry: $entry; each variant keeps its own environment and model directory."
if ($DescribeOnly) { return }
& (Join-Path $PSScriptRoot $entry) -BatchSize $BatchSize -Steps $Steps -EvalEnvs $EvalEnvs
