# BUILD.ps1 -- build nep.exe, gpumd.exe and gpumd_replica.exe from this source tree.
#
# Everything is discovered at run time; nothing is hard-coded to a machine:
#   * nvcc        : PATH, then $env:CUDA_PATH\bin, then the usual install roots
#   * MSVC env    : vswhere -> vcvars64.bat (needed by nvcc's host compiler)
#   * arch        : -Arch, else nvidia-smi compute capability, else sm_60
#
# Exit code is 0 only if all three executables exist afterwards.
#
# usage:
#   .\BUILD.ps1                       # auto arch, build all
#   .\BUILD.ps1 -Arch sm_89 -Jobs 6
#   .\BUILD.ps1 -Clean                # make clean first
param(
    [string]$Arch = "",
    [int]$Jobs = 0,
    [switch]$Clean,
    [string]$Log = "build.log"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Src = Join-Path $Root "src"
if (-not (Test-Path (Join-Path $Src "makefile"))) { throw "src/makefile not found under $Root" }

function Find-Nvcc {
    $c = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    if ($env:CUDA_PATH) {
        $p = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        if (Test-Path $p) { return $p }
    }
    foreach ($r in @("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                     "C:\CUDA")) {
        if (Test-Path $r) {
            $hit = Get-ChildItem $r -Directory | Sort-Object Name -Descending |
                   ForEach-Object { Join-Path $_.FullName "bin\nvcc.exe" } |
                   Where-Object { Test-Path $_ } | Select-Object -First 1
            if ($hit) { return $hit }
        }
    }
    throw "nvcc not found. Install the CUDA Toolkit and/or add it to PATH."
}

function Find-VcVars {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $p = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($p) {
            $b = Join-Path $p "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $b) { return $b }
        }
    }
    foreach ($p in @("C:\Program Files\Microsoft Visual Studio\2022\Community",
                     "C:\Program Files\Microsoft Visual Studio\2022\Professional",
                     "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
                     "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community")) {
        $b = Join-Path $p "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $b) { return $b }
    }
    return $null
}

function Detect-Arch {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return "sm_60" }
    try {
        $cc = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null |
               Select-Object -First 1).Trim()
        if ($cc -match "^(\d+)\.(\d+)$") { return "sm_$($Matches[1])$($Matches[2])" }
    } catch { }
    return "sm_60"
}

$nvcc = Find-Nvcc
$vcvars = Find-VcVars
if (-not $Arch) { $Arch = Detect-Arch }
if ($Jobs -le 0) { $Jobs = [Math]::Max(2, [Environment]::ProcessorCount - 2) }

$make = (Get-Command make -ErrorAction SilentlyContinue).Source
if (-not $make) {
    throw "GNU make not found in PATH. Install make (e.g. via MSYS2/GnuWin32) or run cmake instead."
}

Write-Host "== GPUMD-spin V3 build =="
Write-Host "  source : $Src"
Write-Host "  nvcc   : $nvcc"
Write-Host "  vcvars : $vcvars"
Write-Host "  arch   : $Arch"
Write-Host "  make   : $make"
Write-Host "  jobs   : $Jobs"

$bat = Join-Path $env:TEMP "gpumd_spin_build_$PID.bat"
$lines = @("@echo off")
if ($vcvars) { $lines += "call `"$vcvars`" >nul 2>&1" }
$lines += "cd /d `"$Src`""
if ($Clean) { $lines += "make OS=Windows_NT clean" }
$lines += "make OS=Windows_NT CUDA_ARCH=-arch=$Arch -j $Jobs gpumd"
$lines += "if errorlevel 1 exit /b 10"
$lines += "make OS=Windows_NT CUDA_ARCH=-arch=$Arch -j $Jobs nep"
$lines += "if errorlevel 1 exit /b 11"
$lines += "make OS=Windows_NT CUDA_ARCH=-arch=$Arch gpumd_replica"
$lines += "if errorlevel 1 exit /b 12"
$lines += "exit /b 0"
Set-Content -Path $bat -Value $lines -Encoding ASCII

$sw = [Diagnostics.Stopwatch]::StartNew()
& cmd /c "`"$bat`" 2>&1" | Tee-Object -FilePath (Join-Path $Root $Log)
$rc = $LASTEXITCODE
$sw.Stop()
Remove-Item $bat -ErrorAction SilentlyContinue

Write-Host "`nbuild exit code: $rc   elapsed: $([math]::Round($sw.Elapsed.TotalSeconds,1)) s"
$exes = @("gpumd.exe", "nep.exe", "gpumd_replica.exe")
$missing = @()
foreach ($e in $exes) {
    $p = Join-Path $Src $e
    if (Test-Path $p) {
        $h = (Get-FileHash $p -Algorithm SHA256).Hash.ToLower()
        $size = (Get-Item $p).Length
        Write-Host ("  {0,-20} {1}  {2} bytes" -f $e, $h, $size)
    } else {
        $missing += $e
    }
}
if ($rc -ne 0 -or $missing.Count -gt 0) {
    Write-Error ("build failed: rc=$rc missing=" + ($missing -join ",") + " (see $Log)")
    exit 1
}
Write-Host "== BUILD OK =="
exit 0
