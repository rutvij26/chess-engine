@echo off
echo Chess Engine Launcher
echo ====================
echo.
echo Choose an option:
echo 1. Interactive Mode (play against the engine)
echo 2. UCI Mode (for chess GUIs)
echo 3. Demo Mode (see engine capabilities)
echo 4. Test Mode (run test suite)
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Starting Interactive Mode...
    python interactive.py
) else if "%choice%"=="2" (
    echo Starting UCI Mode...
    echo The engine is now running in UCI mode.
    echo You can connect it to chess GUIs like Arena, Fritz, or Lichess.
    echo Press Ctrl+C to stop.
    python uci_handler.py
) else if "%choice%"=="3" (
    echo Starting Demo Mode...
    python demo.py
) else if "%choice%"=="4" (
    echo Starting Test Mode...
    python test_engine.py
) else if "%choice%"=="5" (
    echo Goodbye!
    exit /b
) else (
    echo Invalid choice. Please run the script again.
    pause
)
