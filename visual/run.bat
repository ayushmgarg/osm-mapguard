@echo off
rem MapGuard review console - double-click, or run: visual\run.bat   (offline: visual\run.bat --offline)
rem Uses the Python install behind visual\.venv directly, so it works from any terminal.
setlocal
cd /d "%~dp0.."
if not exist "visual\.venv\pyvenv.cfg" (
  echo The environment is missing. Set it up once with:
  echo   uv venv --python 3.12 visual\.venv
  echo   uv pip install --python visual\.venv -r visual\requirements.txt
  pause
  exit /b 1
)
set "PYHOME="
for /f "usebackq tokens=1,* delims== " %%a in ("visual\.venv\pyvenv.cfg") do if /i "%%a"=="home" set "PYHOME=%%b"
if not exist "%PYHOME%\python.exe" (
  echo Could not find Python at "%PYHOME%". Recreate the environment:
  echo   uv venv --python 3.12 visual\.venv
  echo   uv pip install --python visual\.venv -r visual\requirements.txt
  pause
  exit /b 1
)
set "PYTHONPATH=%CD%\visual\.venv\Lib\site-packages"
echo Starting MapGuard on http://127.0.0.1:8765 - leave this window open; close it to stop.
start "" /b cmd /c "timeout /t 8 /nobreak >nul 2>nul & start http://127.0.0.1:8765"
"%PYHOME%\python.exe" visual\server.py %*
pause
