$ErrorActionPreference = 'Stop'
$pythonExecutable = (Get-Command python -ErrorAction Stop).Source
$runtimeDirectory = Join-Path $PSScriptRoot '.runtime'
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
$runKey = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff')
$standardOutput = Join-Path $runtimeDirectory ('preview-' + $runKey + '.out.log')
$standardError = Join-Path $runtimeDirectory ('preview-' + $runKey + '.err.log')
# All native arguments are fixed, whitespace-free tokens. The directory is a
# separate Start-Process field, so it is never composed into a shell command.
$process = Start-Process -FilePath $pythonExecutable -ArgumentList @('serve.py', '--no-browser', '--port', '0') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $standardOutput -RedirectStandardError $standardError -PassThru
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if (Test-Path -LiteralPath $standardOutput) {
        $lines = Get-Content -LiteralPath $standardOutput
        $ready = $lines | Where-Object { $_.StartsWith('UI_PREVIEW_READY ') } | Select-Object -First 1
        if ($ready) {
            Write-Output $ready
            Write-Output ('UI_PREVIEW_PID ' + $process.Id)
            Write-Output ('UI_PREVIEW_STDOUT ' + $standardOutput)
            Write-Output ('UI_PREVIEW_STDERR ' + $standardError)
            # Keep the launcher alive: managed execution may reap detached child
            # processes when its parent session exits.
            $process.WaitForExit()
            exit $process.ExitCode
        }
    }
    $process.Refresh()
    if ($process.HasExited) { throw ('Preview process exited; inspect ' + $standardError) }
    Start-Sleep -Milliseconds 100
}
throw ('Preview did not report readiness; inspect ' + $standardError)
