@echo off
rem Imagent launcher for Windows: asks whether to search, scan and index, retry images that could not be indexed,
rem re-embed images, or uninstall, then runs it.
rem Search and scan run inside .venv. The uninstaller does not: it deletes .venv, so it picks its own Python.
rem Usage: run.bat   (double-click it, or run it in a terminal)
setlocal
set "DOUBLE_CLICKED="
echo %cmdcmdline% | find /i "%~nx0" >nul && set "DOUBLE_CLICKED=1"
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

echo What would you like to do?
echo   1) Search your images
echo   2) Scan folders and index images
echo   3) Try again the images that could not be indexed
echo   4) Re-embed images (after the embedding model changed)
echo %ESC%[31m  5) Uninstall Imagent%ESC%[0m
choice /c 12345 /n /m "Choose 1, 2, 3, 4 or 5: "
if errorlevel 5 goto uninstall
set "SCRIPT=search\search.py"
if errorlevel 2 set "SCRIPT=scan_and_index\scan_and_index.py"
if errorlevel 3 set "SCRIPT=retry_failed\retry_failed.py"
if errorlevel 4 set "SCRIPT=reembed\reembed.py"

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
