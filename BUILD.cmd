@echo off
rem BUILD.cmd -- wrapper that bypasses the PowerShell execution policy.
rem Use this when PowerShell says "running scripts is disabled on this system".
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0BUILD.ps1" %*
exit /b %errorlevel%
