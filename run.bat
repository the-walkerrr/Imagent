@echo off
rem Imagent launcher for Windows: asks whether to search, scan and index, or uninstall, then runs it.
rem Search and scan run inside .venv. The uninstaller does not: it deletes .venv, so it picks its own Python.
rem Usage: run.bat   (double-click it, or run it in a terminal)
setlocal
set "DOUBLE_CLICKED="
echo %cmdcmdline% | find /i "%~nx0" >nul && set "DOUBLE_CLICKED=1"
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

echo What would you like to do?
echo   1) Search your images
echo   2) Scan folders and index images
echo %ESC%[31m  3) Uninstall Imagent%ESC%[0m
choice /c 123 /n /m "Choose 1, 2 or 3: "
if errorlevel 3 goto uninstall
set "SCRIPT=search\search.py"
if errorlevel 2 set "SCRIPT=scan_and_index\scan_and_index.py"

if exist "%~dp0.venv\Scripts\activate.bat" goto run

echo The .venv folder was not found. Run the installer first: install.bat
set "CODE=1"
goto finish

:run
call "%~dp0.venv\Scripts\activate.bat"
python "%~dp0%SCRIPT%"
set "CODE=%errorlevel%"
goto finish

:uninstall
call "%~dp0uninstall\uninstall.bat"
set "CODE=%errorlevel%"

:finish
if defined DOUBLE_CLICKED pause
exit /b %CODE%
