@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  EDISON Node Agent — Windows Quick Installer
REM  Run this script on your engineering laptop to set up the agent.
REM ═══════════════════════════════════════════════════════════════════

echo.
echo  ███████╗██████╗ ██╗███████╗ ██████╗ ███╗   ██╗
echo  ██╔════╝██╔══██╗██║██╔════╝██╔═══██╗████╗  ██║
echo  █████╗  ██║  ██║██║███████╗██║   ██║██╔██╗ ██║
echo  ██╔══╝  ██║  ██║██║╚════██║██║   ██║██║╚██╗██║
echo  ███████╗██████╔╝██║███████║╚██████╔╝██║ ╚████║
echo  ╚══════╝╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
echo                  Node Agent Setup
echo.

REM Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not on PATH.
    echo Download Python 3.10+ from https://python.org/downloads
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/4] Python found:
python --version
echo.

REM Install dependencies
echo [2/4] Installing dependencies...
pip install requests psutil
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] pip install had issues, but continuing...
)
echo.

REM Optional: pywin32 for Rhino COM automation
echo [3/4] Installing Rhino COM support (pywin32)...
pip install pywin32
echo.

REM Prompt for EDISON server IP
echo [4/4] Configuration
echo.
set /p EDISON_SERVER="Enter your EDISON server IP address: "
set /p NODE_NAME="Enter a name for this node (e.g. Engineering-Laptop): "

if "%NODE_NAME%"=="" set NODE_NAME=%COMPUTERNAME%

echo.
echo ═══════════════════════════════════════════════════
echo  Setup Complete!
echo ═══════════════════════════════════════════════════
echo.
echo  To start the agent, run:
echo.
echo    python edison_node_agent.py --server %EDISON_SERVER% --name "%NODE_NAME%" --role cad
echo.
echo  TIP: For Rhino integration, start Rhino 7 FIRST, then run the agent.
echo.
echo  To auto-start at login:
echo    1. Press Win+R, type: shell:startup
echo    2. Create a shortcut to:
echo       python "%CD%\edison_node_agent.py" --server %EDISON_SERVER% --name "%NODE_NAME%" --role cad
echo.

REM Ask if they want to start now
set /p START_NOW="Start the agent now? (y/n): "
if /i "%START_NOW%"=="y" (
    echo.
    echo Starting EDISON Node Agent...
    echo Press Ctrl+C to stop.
    echo.
    python edison_node_agent.py --server %EDISON_SERVER% --name "%NODE_NAME%" --role cad
)

pause
