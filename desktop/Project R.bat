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

rem pythonw: the same Python without a console window.
set "PYW=%PY:python.exe=pythonw.exe%"
set "PYW=%PYW:\py.exe=\pyw.exe%"
if not exist "%PYW%" set "PYW=%PY%"
echo Starting Project R...
start "" "%PYW%" "desktop\app.py" %*
exit /b 0

:no_python
echo No Python found. Make the project's venv first, in this folder:
echo     python -m venv .venv
echo     .venv\Scripts\pip install -r requirements.txt -r requirements-webapp.txt -r requirements-desktop.txt
echo.
pause
exit /b 1

:missing
echo.
echo Install what is missing into this Python, then double-click me again:
echo     "%PY%" -m pip install -r requirements.txt -r requirements-webapp.txt -r requirements-desktop.txt
echo.
pause
exit /b 1
