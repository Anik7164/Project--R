@echo off
rem Project R as a desktop app. Double-click me.
rem   "Project R.bat" --share     also lets phones on the Wi-Fi join (iPhone app, viewers)
cd /d "%~dp0.."
set "PY=pythonw"
if exist ".venv\Scripts\pythonw.exe" set "PY=.venv\Scripts\pythonw.exe"
start "" "%PY%" "desktop\app.py" %*
