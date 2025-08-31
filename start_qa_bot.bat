@echo off
echo.
echo ========================================
echo    RAG QA Bot - Generative AI Apps
echo ========================================
echo.
echo Starting the QA Bot...
echo.
echo This will:
echo 1. Check your system configuration
echo 2. Launch the web interface
echo 3. Open your browser automatically
echo.
echo Press any key to continue...
pause >nul

echo.
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo.
echo Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo Starting QA Bot...
echo.
echo The web interface will open in your browser at:
echo http://localhost:7860
echo.
echo To stop the bot, press Ctrl+C in this window
echo.

python main_qa_bot.py

echo.
echo QA Bot stopped.
pause
