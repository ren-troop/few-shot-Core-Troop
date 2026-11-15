@echo off
chcp 65001 > nul
pushd "%~dp0"

if exist "outputs_seed42" rmdir /s /q "outputs_seed42"

set PYTHON=python
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe

"%PYTHON%" run_all.py --download ^
  --seed 42 ^
  --episodes 600 ^
  --eval-episodes 200 ^
  --meta-batch 4 ^
  --val-every 100 ^
  --val-episodes 50 ^
  --inner-lr 0.4 ^
  --outer-lr 0.001 ^
  --initial-l2 0.001 ^
  --inner-steps 1 ^
  --n-way 5 ^
  --k-shot 1 ^
  --q-query 5 ^
  --pretrain-epochs 30 ^
  --fixed2-steps 2 ^
  --fixed2-lr 0.05 ^
  --output-root outputs_seed42 ^
  --plots

if errorlevel 1 goto failed
echo All methods completed. Results are in outputs_seed42.
popd
pause
exit /b 0

:failed
echo Failed. Copy the full error before closing.
popd
pause
exit /b 1
