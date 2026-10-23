@echo off
rem MapGuard review console - double-click, or run: visual\run.bat   (offline: visual\run.bat --offline)
rem Python: visual\.runtime\python.exe (a private copy inside the project, git-ignored), else the
rem install visual\.venv was built from. Packages: visual\.venv\Lib\site-packages.
setlocal
cd /d "%~dp0.."
set "PY=%CD%\visual\.runtime\python.exe"
if exist "%PY%" goto have_python
set "PYHOME="
if exist "visual\.venv\pyvenv.cfg" for /f "usebackq tokens=1,* delims== " %%a in ("visual\.venv\pyvenv.cfg") do if /i "%%a"=="home" set "PYHOME=%%b"
set "PY=%PYHOME%\python.exe"
if exist "%PY%" goto have_python
echo No Python found for the console. Set it up once (from the repository root):
echo   uv venv --python 3.12 visual\.venv
echo   uv pip install --python visual\.venv -r visual\requirements.txt
echo then copy the Python install named in visual\.venv\pyvenv.cfg ("home") to visual\.runtime
pause
exit /b 1

:have_python
if not exist "visual\.venv\Lib\site-packages" (
  echo The packages are missing: uv pip install --python visual\.venv -r visual\requirements.txt
  pause
  exit /b 1
)
set "PYTHONPATH=%CD%\visual\.venv\Lib\site-packages"
set "PYTHONHOME="
echo Starting MapGuard on http://127.0.0.1:8765 - leave this window open; close it to stop.
echo Loading takes about 20 seconds; the browser opens when the server is ready.
start "" /b cmd /c "timeout /t 20 /nobreak >nul 2>nul & start http://127.0.0.1:8765"
"%PY%" visual\server.py %*
pause
