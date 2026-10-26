@echo off
REM Double-click to start the Football Predictor.
cd /d "%~dp0"
where py >nul 2>nul && (py -3 -m football_predictor %*) || (python -m football_predictor %*)
if errorlevel 1 pause
