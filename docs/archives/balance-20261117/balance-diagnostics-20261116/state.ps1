$ErrorActionPreference = 'Stop'
$state = Invoke-RestMethod -Uri 'http://127.0.0.1:8095/state' -TimeoutSec 12
$training = $state.training
$training | Select-Object mode, active, phase, stage, run_id, steps, checkpoint, script, stop_reason, evaluation_scope | ConvertTo-Json -Depth 4
