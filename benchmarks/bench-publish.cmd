@echo off
REM bench-publish.cmd — Wrapper that runs bench-publish.ps1 without requiring
REM a system-wide ExecutionPolicy change. Run from an admin command prompt.
REM
REM Usage:
REM   benchmarks\bench-publish.cmd                              — auto-detects base clock
REM   benchmarks\bench-publish.cmd -Rounds 20                   — more rounds
REM   benchmarks\bench-publish.cmd -Clock 1800                  — override clock frequency
REM   benchmarks\bench-publish.cmd -Output benchmarks/report.txt — save report to file

REM UNC paths (\\wsl.localhost\...) are not supported by cmd.exe.
REM Push to a temp drive letter so cmd.exe is happy, then call the .ps1.
pushd "%~dp0" 2>nul || cd /d "%TEMP%"
powershell -ExecutionPolicy Bypass -File "%~dp0bench-publish.ps1" %*
popd 2>nul
