@echo off
echo ========================================
echo   AgriSmart AI Platform - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
echo.

REM Install dependencies if needed
echo Checking dependencies...
pip install -q -r requirements.txt
echo Dependencies installed!
echo.

REM Initialize database
echo Initializing database...
python utils\database.py
echo.

REM Run the application
echo Starting Flask application...
echo.
echo ====================================================================================
echo   Application will be available at: http://localhost:5000
echo   Press Ctrl+C to stop the server
echo ====================================================================================
echo.

python app.py
