@echo off
REM Runs the daily snapshot collector. Windows Task Scheduler calls this file once a day.
REM It moves into the project folder, runs the script with the project's own Python,
REM and adds the output to data\history\collector.log so you can check it later.

cd /d "%~dp0.."
if not exist "data\history" mkdir "data\history"
".venv\Scripts\python.exe" scripts\collect_snapshot.py >> "data\history\collector.log" 2>&1
