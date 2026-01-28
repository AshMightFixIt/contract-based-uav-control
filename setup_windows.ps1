# Windows Setup Script
# Run this before running any Python scripts

$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"

Write-Host "✓ Python path set!"
Write-Host "Current directory: $PWD"
Write-Host "PYTHONPATH: $env:PYTHONPATH"
Write-Host ""
Write-Host "Now you can run:"
Write-Host "  python simulation_demo.py"
Write-Host "  python src\contracts\contract_framework.py"
Write-Host "  python src\estimation\contract_ekf.py"
Write-Host "  python src\control\controllers.py"
