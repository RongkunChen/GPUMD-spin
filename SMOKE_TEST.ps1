# SMOKE_TEST.ps1 -- run the smoke tests against an existing build.
param([string]$Bin = "", [string]$Work = "")
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Bin) { $Bin = Join-Path $Root "src" }
$args = @((Join-Path $Root "tests\smoke.py"), "--bin", $Bin,
          "--json", (Join-Path $Root "smoke_result.json"))
if ($Work) { $args += @("--work", $Work) }
python @args
exit $LASTEXITCODE
