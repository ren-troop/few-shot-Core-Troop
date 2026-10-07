@echo off
chcp 65001 > nul
pushd "%~dp0"
for %%S in (42 43 44) do (
  python run_all.py --download --seed %%S --episodes 600 --eval-episodes 200 --meta-batch 4 --val-every 100 --val-episodes 50 --inner-lr 0.4 --outer-lr 0.001 --initial-l2 0.0001 --l2-outer-lr 0.01 --l2-adam-eps 0.00000001 --hidden-channels 32 --plots --output-root outputs_review_seed%%S
  if errorlevel 1 goto failed
)
echo All seeds completed. Run tools\export_results.py from the repository root.
popd
pause
exit /b 0
:failed
echo Failed. No further seeds were launched. Copy the full error before closing.
popd
pause
exit /b 1
