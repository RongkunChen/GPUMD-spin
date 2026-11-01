# INSTALL.ps1 -- check the toolchain, build, and run the smoke tests.
#
# usage:  .\INSTALL.ps1            (build + smoke)
#         .\INSTALL.ps1 -NoSmoke
param([switch]$NoSmoke, [string]$Arch = "", [int]$Jobs = 0)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "== prerequisites =="
foreach ($t in @("nvcc", "make", "nvidia-smi", "python")) {
    $c = Get-Command $t -ErrorAction SilentlyContinue
    Write-Host ("  {0,-12} {1}" -f $t, ($(if ($c) { $c.Source } else { "NOT FOUND" })))
}
& (Join-Path $Root "BUILD.ps1") -Arch $Arch -Jobs $Jobs
if ($LASTEXITCODE -ne 0) { Write-Error "build failed"; exit 1 }

if (-not $NoSmoke) {
    Write-Host "`n== smoke tests =="
    python (Join-Path $Root "tests\smoke.py") --bin (Join-Path $Root "src") `
        --json (Join-Path $Root "smoke_result.json")
    if ($LASTEXITCODE -ne 0) { Write-Error "smoke tests failed"; exit 2 }
}
Write-Host "`n== INSTALL OK =="
exit 0
