@echo off
setlocal EnableExtensions
cd /d "%~dp0"

git lfs version >nul 2>&1
if errorlevel 1 (
  echo Git LFS is required. Install it, run "git lfs install", and retry.
  exit /b 1
)
git lfs pull
if errorlevel 1 exit /b 1

if defined UV_EXE goto uv_ready
for /f "delims=" %%I in ('where uv 2^>nul') do if not defined UV_EXE set "UV_EXE=%%I"
if defined UV_EXE goto uv_ready
if exist "%APPDATA%\Python\Scripts\uv.exe" set "UV_EXE=%APPDATA%\Python\Scripts\uv.exe"

:uv_ready
if not defined UV_EXE (
  echo uv was not found. Install uv or set UV_EXE to its executable path.
  exit /b 1
)

if not exist ".venv\Scripts\weblfp.exe" (
  "%UV_EXE%" sync --locked
  if errorlevel 1 exit /b 1
)

if not defined SKIP_FRONTEND_BUILD (
  where npm >nul 2>&1
  if errorlevel 1 (
    echo npm is required to build the frontend.
    exit /b 1
  )
  pushd frontend
  call npm ci
  if errorlevel 1 exit /b 1
  call npm run build
  if errorlevel 1 exit /b 1
  popd
)

"%UV_EXE%" run --no-sync weblfp
exit /b %ERRORLEVEL%
