function Resolve-RobotMode {
    param([ValidateSet('selected', 'original', 'torque200')][string]$Mode)
    if ($Mode -ne 'selected') { return $Mode }
    $code = 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text())["mode"] if p.exists() else "original")'
    $nativeArgs = @('-d', 'Ubuntu', '--exec', '/usr/bin/python3', '-c', $code,
        '/home/jaccob/projects/tenny-torque200/artifacts/selected-mode.json')
    $value = & wsl.exe @nativeArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw 'Cannot read selected mode; no training was started' }
    $resolved = ($value -join '').Trim()
    if ($resolved -notin @('original', 'torque200')) { throw "Invalid selected mode: $resolved" }
    return $resolved
}
