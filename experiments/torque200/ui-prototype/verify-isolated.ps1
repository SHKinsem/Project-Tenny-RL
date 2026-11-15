$ErrorActionPreference = 'Stop'
$temporaryRoot = Join-Path $PSScriptRoot '.runtime\test-temp'
New-Item -ItemType Directory -Path $temporaryRoot -Force | Out-Null
$env:TMPDIR = $temporaryRoot
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$nodeArguments = @(
    '--test'
    (Join-Path $PSScriptRoot 'mock-engine.test.mjs')
    (Join-Path $PSScriptRoot 'layout.test.mjs')
    (Join-Path $PSScriptRoot 'chart-model.test.mjs')
)
& node @nodeArguments
if ($LASTEXITCODE -ne 0) { throw 'Isolated Node checks failed' }
$pythonArguments = @('-m', 'unittest', 'discover', '-s', $PSScriptRoot, '-p', 'test_preview_server.py', '-v')
& python @pythonArguments
if ($LASTEXITCODE -ne 0) { throw 'Read-only preview server checks failed' }
$syntaxArguments = @('--check', (Join-Path $PSScriptRoot 'app.mjs'))
& node @syntaxArguments
if ($LASTEXITCODE -ne 0) { throw 'UI app syntax check failed' }
Write-Output 'ISOLATED_UI_CHECKS_OK'
