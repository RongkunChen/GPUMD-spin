# run-all.ps1 -- one-shot controller for the GPUMD-spin V3 phases.
#
# Each phase is a python driver under tools/one_shot/.  The controller only
# decides *whether* a phase may run (declared prerequisites, checkpoint state);
# every acceptance decision is made inside the driver and written to
# <phase>_results.json.  A failed phase does not stop the run: independent
# branches continue, which is what the instruction asks for.
#
# usage:
#   .\run-all.ps1                 # from the first non-PASS phase to the end
#   .\run-all.ps1 -From 3         # start at phase 3
#   .\run-all.ps1 -Only 3         # run one phase
param(
    [int]$From = 0,
    [int]$Only = -1,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$ts = Get-Content ".one_shot_dir" -ErrorAction SilentlyContinue
if (-not $ts) { throw "no .one_shot_dir; run tools/one_shot/phase0_freeze.py first" }
$Dir = "evidence/one_shot_$ts"
$Ckpt = "$Dir/checkpoints/checkpoint.json"
$Log = "$Dir/checkpoints/controller.jsonl"

function Log($obj) {
    $line = $obj | ConvertTo-Json -Compress
    Add-Content -Path $Log -Value $line -Encoding utf8
}

# phase -> (script, prerequisites)
$Phases = [ordered]@{
    0 = @{ Script = "tools/one_shot/phase0_freeze.py";   Args = @("$Dir"); Pre = @() }
    1 = @{ Script = "tools/one_shot/phase1_static.py";   Args = @("$Dir"); Pre = @(0) }
    2 = @{ Script = "tools/one_shot/phase2_s05b_r2.py";  Args = @("$Dir"); Pre = @(1) }
    3 = @{ Script = "tools/one_shot/phase3_s08b.py";     Args = @("$Dir"); Pre = @(1) }
    4 = @{ Script = "tools/one_shot/phase4_train.py";    Args = @("$Dir"); Pre = @(1) }
    5 = @{ Script = "tools/one_shot/phase5_s08c.py";     Args = @("$Dir"); Pre = @(3, 4) }
    6 = @{ Script = "tools/one_shot/phase6_s09.py";      Args = @("$Dir"); Pre = @(2, 3) }
    7 = @{ Script = "tools/one_shot/phase7_magnon.py";   Args = @("$Dir"); Pre = @(4) }
    8 = @{ Script = "tools/one_shot/phase8_final.py";    Args = @("$Dir"); Pre = @() }
}

function Status($n) {
    if (-not (Test-Path $Ckpt)) { return "PENDING" }
    $c = Get-Content $Ckpt -Raw | ConvertFrom-Json
    $p = $c.phases."$n"
    if ($null -eq $p) { return "PENDING" }
    return $p.status
}

$order = $Phases.Keys | Sort-Object
foreach ($n in $order) {
    if ($Only -ge 0 -and $n -ne $Only) { continue }
    if ($n -lt $From) { continue }
    $script = $Phases[$n].Script
    if (-not (Test-Path $script)) {
        Write-Host "[phase $n] SKIPPED (no driver: $script)"
        Log @{ phase = $n; event = "skipped"; reason = "missing driver"; script = $script }
        continue
    }
    $blocked = @()
    foreach ($p in $Phases[$n].Pre) {
        $s = Status $p
        if ($s -ne "PASS") { $blocked += "phase $p = $s" }
    }
    if ($blocked.Count -gt 0 -and -not $Force) {
        Write-Host "[phase $n] BLOCKED by prerequisites: $($blocked -join ', ')"
        Log @{ phase = $n; event = "blocked"; reason = ($blocked -join "; ") }
        continue
    }
    Write-Host "[phase $n] running $script"
    Log @{ phase = $n; event = "start"; script = $script; argv = $Phases[$n].Args }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & python $script @($Phases[$n].Args)
    $rc = $LASTEXITCODE
    $sw.Stop()
    $st = Status $n
    Write-Host "[phase $n] rc=$rc status=$st elapsed=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
    Log @{ phase = $n; event = "end"; rc = $rc; status = $st;
           seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1) }
}
Write-Host "done. checkpoint: $Ckpt"
