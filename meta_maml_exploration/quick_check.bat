@echo off
chcp 65001 > nul
pushd "%~dp0"
if exist "outputs_quick_check_clamp" rmdir /s /q "outputs_quick_check_clamp"
python run_all.py --download --episodes 3 --eval-episodes 3 --meta-batch 1 --val-every 1 --val-episodes 3 --meta-sgd-mode clamp --plots --output-root outputs_quick_check_clamp
if errorlevel 1 goto failed
echo Quick check completed.
popd
pause
exit /b 0
:failed
echo Failed. Copy the full error. Use a new output-root if this directory already exists.
popd
pause
exit /b 1
