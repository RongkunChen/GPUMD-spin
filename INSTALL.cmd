@echo off
rem INSTALL.cmd -- build + smoke, without touching the machine-wide execution policy.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0INSTALL.ps1" %*
exit /b %errorlevel%
