# resume.ps1 -- continue the one-shot run from the last non-PASS phase.
#
# Reads evidence/one_shot_<ts>/checkpoints/checkpoint.json, finds the first phase
# whose status is not PASS, and re-runs from there.  Phases already PASS are not
# recomputed (their evidence is hash-registered in the checkpoint).
#
# usage:
#   .\resume.ps1
#   .\resume.ps1 -Force       # ignore the checkpoint and rerun from phase 0
param([switch]$Force)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$ts = Get-Content ".one_shot_dir"
$Ckpt = "evidence/one_shot_$ts/checkpoints/checkpoint.json"
if (-not (Test-Path $Ckpt)) { throw "checkpoint missing: $Ckpt" }

$c = Get-Content $Ckpt -Raw | ConvertFrom-Json
$start = 0
if (-not $Force) {
    foreach ($n in 0..8) {
        $p = $c.phases."$n"
        if ($null -eq $p -or $p.status -ne "PASS") { $start = $n; break }
        $start = 9
    }
}
if ($start -gt 8) {
    Write-Host "all phases PASS according to the checkpoint; nothing to resume."
    Write-Host "use -Force to rerun from phase 0 anyway."
    exit 0
}
Write-Host "resuming from phase $start"
& "$Root/run-all.ps1" -From $start
