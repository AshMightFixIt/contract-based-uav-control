@echo off
REM Windows Setup Script (for Command Prompt)
REM Run this before running any Python scripts

set PYTHONPATH=%CD%\src;%PYTHONPATH%

echo ✓ Python path set!
echo Current directory: %CD%
echo PYTHONPATH: %PYTHONPATH%
echo.
echo Now you can run:
echo   python simulation_demo.py
echo   python src\contracts\contract_framework.py
echo   python src\estimation\contract_ekf.py
echo   python src\control\controllers.py
