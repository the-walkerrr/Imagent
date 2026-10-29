@echo off
rem Imagent launcher for Windows: finds Python 3.9 or newer and runs install.py with it.
rem It installs nothing; if Python is missing it says how to get it.
rem Usage: install.bat   (double-click it, or run it in a terminal)
setlocal
set "SCRIPT=%~dp0install.py"
set "DOUBLE_CLICKED="
echo %cmdcmdline% | find /i "%~nx0" >nul && set "DOUBLE_CLICKED=1"

py -3 -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>&1
if not errorlevel 1 goto use_py

python -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>&1
if not errorlevel 1 goto use_python

echo Imagent needs Python 3.9 or newer, and none was found on this computer.
echo Install one, then run this again: install.bat
echo.
echo   Open a terminal and run:  winget install Python.Python.3.12
echo   or download it from https://www.python.org/downloads/
echo   (tick "Add python.exe to PATH" in the installer). Then open a NEW terminal window.
set "CODE=1"
goto finish

:use_py
py -3 "%SCRIPT%" %*
set "CODE=%errorlevel%"
goto finish

:use_python
python "%SCRIPT%" %*
set "CODE=%errorlevel%"

:finish
if defined DOUBLE_CLICKED pause
exit /b %CODE%
