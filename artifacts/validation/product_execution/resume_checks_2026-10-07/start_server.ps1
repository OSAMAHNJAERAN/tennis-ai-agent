$ErrorActionPreference = 'Stop'
$resumeWorkspace = (Resolve-Path -LiteralPath '.').Path
$resumeUnit = Join-Path $resumeWorkspace 'artifacts/validation/product_execution/resume_checks_2026-10-07'
$resumePython = 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe'
& $resumePython (Join-Path $resumeUnit 'check_launch.py')
if ($LASTEXITCODE -ne 0) { throw 'Read-only launch check failed; preserve runtime' }
$resumeOutput = Join-Path $resumeUnit 'server_stdout.log'
$resumeError = Join-Path $resumeUnit 'server_stderr.log'
$resumeReceipt = Join-Path $resumeUnit 'server_launch.json'
if ((Test-Path -LiteralPath $resumeOutput) -or (Test-Path -LiteralPath $resumeError) -or (Test-Path -LiteralPath $resumeReceipt)) {
    throw 'Preserve the consumed launch'
}
$resumeServer = Start-Process -FilePath $resumePython -ArgumentList 'scripts/run/serve_product.py' -WorkingDirectory $resumeWorkspace -WindowStyle Hidden -RedirectStandardOutput $resumeOutput -RedirectStandardError $resumeError -PassThru
$resumeLaunch = [ordered]@{ launchedAt=(Get-Date).ToUniversalTime().ToString('o'); pid=$resumeServer.Id; processStartedAt=$resumeServer.StartTime.ToUniversalTime().ToString('o'); executable=$resumePython; argument='scripts/run/serve_product.py'; workingDirectory=$resumeWorkspace; windowStyle='Hidden'; previousRuntimeStopped=$false }
$resumeLaunch | ConvertTo-Json | Set-Content -LiteralPath $resumeReceipt -Encoding UTF8
$resumeLaunch | ConvertTo-Json -Compress
