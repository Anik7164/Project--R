@echo off
setlocal EnableExtensions
rem Project R as a desktop app. Double-click me.
rem   "Project R.bat" --share     also lets phones on the Wi-Fi join (iPhone app, viewers)
rem
rem It finds the project's Python (.venv or venv in the project folder; else
rem the Python on PATH), checks that the web app is installed in it, and then
rem starts the app without a console window. If something is missing, this
rem window stays open and says what.
title Project R
cd /d "%~dp0.."

set "PY="
rem The offline setup file brings its own Python in runtime\.
if exist "runtime\python.exe" set "PY=%CD%\runtime\python.exe"
for %%V in (".venv" "venv" "env" "..\.venv" "..\venv") do (
  if not defined PY if exist "%%~V\Scripts\python.exe" set "PY=%%~fV\Scripts\python.exe"
)
if not defined PY (
  rem The python.exe in WindowsApps only opens the Microsoft Store: skip it.
  for /f "delims=" %%W in ('where python 2^>nul ^| findstr /v /i "WindowsApps"') do (
    if not defined PY set "PY=%%W"
  )
)
if not defined PY (
  for /f "delims=" %%W in ('where py 2^>nul') do if not defined PY set "PY=%%W"
)
if not defined PY goto no_python

"%PY%" "desktop\app.py" --check
if errorlevel 1 goto missing

rem "Project R.bat" --console  runs it in this window, so every message and error shows.
if /i "%~1"=="--console" (
  shift
  echo Starting Project R in this window. Close the app window to stop.
  "%PY%" "desktop\app.py" %2 %3 %4 %5
  echo.
  echo Project R stopped.
  pause
  exit /b 0
)

rem pythonw: the same Python without a console window.
set "PYW=%PY:python.exe=pythonw.exe%"
set "PYW=%PYW:\py.exe=\pyw.exe%"
if not exist "%PYW%" set "PYW=%PY%"
echo Starting Project R... the window opens in a few seconds.
echo If it does not, run:  "desktop\Project R.bat" --console
echo and the problem shows here. Log: .cache\desktop\app.log
start "" "%PYW%" "desktop\app.py" %*
exit /b 0

:no_python
echo No Python found. Make the project's venv first, in this folder:
echo     python -m venv .venv
echo     .venv\Scripts\pip install -r requirements.txt
echo.
pause
exit /b 1

:missing
echo.
echo Install what is missing into this Python, then double-click me again:
echo     "%PY%" -m pip install -r requirements.txt
echo.
pause
exit /b 1
