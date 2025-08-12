@echo off
REM ====== CONFIG ======
set PORT=5000
set REGION=ap
REM 如果用 Conda，填环境名；不用就留空
set CONDA_ENV=aps360
REM ====================

echo.
echo [1/3] Starting Flask on port %PORT%...
if not "%CONDA_ENV%"=="" (
  start "Flask" cmd /k "conda activate %CONDA_ENV% && python app.py"
) else (
  start "Flask" cmd /k "python app.py"
)

echo [2/3] Starting ngrok tunnel...
REM 稍等 Flask 起服务
timeout /t 3 >nul
start "ngrok" cmd /k "ngrok http %PORT% --region=%REGION%"

REM 等 ngrok 起來并提供 4040 API，然后读取 public_url
echo [3/3] Fetching public URL from ngrok...
set PUBURL=
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command ^
  "$ErrorActionPreference='SilentlyContinue'; ^
   for($i=0;$i -lt 30;$i++){ ^
     try { $u=(Invoke-RestMethod http://127.0.0.1:4040/api/tunnels).tunnels ^
            | ? { $_.proto -eq 'https' } ^
            | select -exp public_url; } catch{} ^
     if($u){ $u; break } ^
     Start-Sleep -Seconds 1 }"`) do set PUBURL=%%i

if "%PUBURL%"=="" (
  echo Failed to get ngrok public URL. Open http://127.0.0.1:4040 to debug.
  goto :eof
)

echo.
echo =================  DEMO LINK  =================
echo %PUBURL%
echo ===============================================
echo.

REM 复制到剪贴板，并用默认浏览器打开
powershell -NoProfile -Command "Set-Clipboard '%PUBURL%'; Start-Process '%PUBURL%'"
echo (The URL has been copied to your clipboard.)
echo Done. Keep both windows (Flask, ngrok) open during the demo.
