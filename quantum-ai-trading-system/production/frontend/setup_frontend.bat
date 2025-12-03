@echo off
REM Quantum AI Cockpit - Frontend Setup Script
REM This script installs all npm dependencies

echo ========================================
echo Quantum AI Cockpit - Frontend Setup
echo ========================================
echo.

REM Change to frontend directory
cd /d "%~dp0"

REM Check if node_modules exists
if exist "node_modules" (
    echo node_modules found. Updating dependencies...
) else (
    echo Installing dependencies...
)

echo.
echo Installing npm packages...
call npm install

echo.
echo ========================================
echo Frontend setup complete!
echo ========================================
echo.
echo To start the frontend dev server, run:
echo   npm run dev
echo.
pause

