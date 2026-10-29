@echo off
rem Imagent search launcher for Windows: activates .venv and runs search.py.
rem Usage: search.bat [query]   (double-click it, or run it in a terminal)
setlocal
set "DOUBLE_CLICKED="
echo %cmdcmdline% | find /i "%~nx0" >nul && set "DOUBLE_CLICKED=1"

if exist "%~dp0.venv\Scripts\activate.bat" goto run

echo The .venv folder was not found. Run the installer first: install.bat
set "CODE=1"
goto finish

:run
call "%~dp0.venv\Scripts\activate.bat"
python "%~dp0search.py" %*
set "CODE=%errorlevel%"

:finish
if defined DOUBLE_CLICKED pause
exit /b %CODE%
