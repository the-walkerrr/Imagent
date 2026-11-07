@echo off
rem Imagent launcher for Windows: opens a menu (arrow keys and Enter) to search, scan and index, see the images that
rem could not be indexed, re-embed images (only offered when the embedding model changed), or uninstall.
rem The menu and the commands run inside .venv. The uninstaller does not: it deletes .venv, so it picks its own Python.
rem If Imagent is not installed yet, it offers to run the installer first.
rem Usage: run.bat   (double-click it, or run it in a terminal)
setlocal
set "DOUBLE_CLICKED="
echo %cmdcmdline% | find /i "%~nx0" >nul && set "DOUBLE_CLICKED=1"
rem The same number as in menu\menu.py
set "UNINSTALL_EXIT_CODE=75"

if exist "%~dp0.venv\Scripts\activate.bat" goto run

echo Imagent is not installed yet (the .venv folder was not found).
choice /c YN /n /m "Install it now? (Y/N): "
if errorlevel 2 goto declined
call "%~dp0install.bat"
set "CODE=%errorlevel%"
if not "%CODE%"=="0" goto finish
if exist "%~dp0.venv\Scripts\activate.bat" goto run
echo The installer finished but the .venv folder is still missing.
set "CODE=1"
goto finish

:declined
echo Nothing was installed. Run the installer any time: install.bat
echo To remove Imagent without it: uninstall\uninstall.bat
set "CODE=1"
goto finish

:run
rem Its own scope, so .venv is not active when the uninstaller starts
setlocal
call "%~dp0.venv\Scripts\activate.bat"
python "%~dp0menu\menu.py"
set "CODE=%errorlevel%"
endlocal & set "CODE=%CODE%"
if not "%CODE%"=="%UNINSTALL_EXIT_CODE%" goto finish
call "%~dp0uninstall\uninstall.bat"
set "CODE=%errorlevel%"

:finish
if defined DOUBLE_CLICKED pause
exit /b %CODE%
