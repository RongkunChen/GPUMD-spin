@echo off
rem SMOKE_TEST.cmd -- same as SMOKE_TEST.ps1, bypassing the execution policy.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0SMOKE_TEST.ps1" %*
exit /b %errorlevel%
