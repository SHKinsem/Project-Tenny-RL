param(
    [ValidateSet('selected', 'original', 'torque200')][string]$Mode = 'selected',
    [ValidateRange(1024, 65535)][int]$Port = 8770
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'tasks/resolve-robot-mode.ps1')
$resolved = Resolve-RobotMode -Mode $Mode
& (Join-Path $PSScriptRoot 'start-torque200-web.ps1') -Mode $resolved -Port $Port
