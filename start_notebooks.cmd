@echo off
setlocal
cd /d "%~dp0"
for /f "delims=" %%P in ('wsl -d Ubuntu-22.04 -- wslpath -a "%CD%"') do set "FLIDS_WSL_PROJECT=%%P"
wsl -d Ubuntu-22.04 -- bash "%FLIDS_WSL_PROJECT%/launch.sh" notebooks
if errorlevel 1 pause
