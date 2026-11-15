@echo off
rem Imagent uninstaller launcher for Windows: finds Python 3.9 or newer and runs the uninstaller with it.
rem The uninstaller shows what will be lost and asks three times before it deletes anything.
rem Usage: uninstall\uninstall.bat   (double-click it, or run it in a terminal)
setlocal
set "DOUBLE_CLICKED="
echo %cmdcmdline% | find /i "%~nx0" >nul && set "DOUBLE_CLICKED=1"

py -3 -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>&1
if not errorlevel 1 goto use_py

python -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>&1
if not errorlevel 1 goto use_python

echo The uninstaller needs Python 3.9 or newer, and none was found on this computer.
echo Install one, or remove Imagent by hand:
echo   - delete the .venv folder in this project and the Imagent data folder (see the README)
echo   - remove the Ollama models Imagent downloaded: ollama rm NAME
set "CODE=1"
goto finish

:use_py
py -3 -B "%~dp0uninstall.py" %*
set "CODE=%errorlevel%"
goto finish

:use_python
python -B "%~dp0uninstall.py" %*
set "CODE=%errorlevel%"

:finish
if defined DOUBLE_CLICKED pause
exit /b %CODE%
