@echo off
rem Double-click to start all of Beacon: containers, gateway, retrieval,
rem worker and the console, then open http://localhost:3000.
rem Docker Desktop must already be running.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tasks.ps1" dev
if errorlevel 1 pause
