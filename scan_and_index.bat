@echo off
rem Imagent scan and index launcher for Windows: activates .venv and runs scan_and_index.py.
rem Usage: scan_and_index.bat [folder]   (double-click it, or run it in a terminal)
setlocal
set "DOUBLE_CLICKED="
echo %cmdcmdline% | find /i "%~nx0" >nul && set "DOUBLE_CLICKED=1"

if exist "%~dp0.venv\Scripts\activate.bat" goto run

echo The .venv folder was not found. Run the installer first: install.bat
set "CODE=1"
goto finish

:run
call "%~dp0.venv\Scripts\activate.bat"
python "%~dp0scan_and_index.py" %*
set "CODE=%errorlevel%"

:finish
if defined DOUBLE_CLICKED pause
exit /b %CODE%
