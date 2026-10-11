@echo off
chcp 65001 > nul
pushd "%~dp0"
rem Remove the exact prior experiment roots so this run fully replaces their results.
for %%S in (42 43 44) do if exist "outputs_review_seed%%S" rmdir /s /q "outputs_review_seed%%S"
if exist "outputs_quick_check" rmdir /s /q "outputs_quick_check"
if exist "outputs_quick_check_clamp" rmdir /s /q "outputs_quick_check_clamp"
if exist "..\results\legacy_v1" rmdir /s /q "..\results\legacy_v1"
if exist "..\results\review_v2" rmdir /s /q "..\results\review_v2"
if exist "..\results\verification\smoke_checks.json" del /q "..\results\verification\smoke_checks.json"
if exist "..\results\verification\FINAL_PACKAGE_AUDIT.md" del /q "..\results\verification\FINAL_PACKAGE_AUDIT.md"
if exist "..\results\verification\VALIDATION.md" del /q "..\results\verification\VALIDATION.md"
if exist "outputs_review_feedback.zip" del /q "outputs_review_feedback.zip"
if exist "outputs_review_feedback_clamp.zip" del /q "outputs_review_feedback_clamp.zip"
for %%S in (42 43 44) do (
  python run_all.py --download --seed %%S --episodes 600 --eval-episodes 200 --meta-batch 4 --val-every 100 --val-episodes 50 --inner-lr 0.4 --outer-lr 0.001 --initial-l2 0.0001 --l2-outer-lr 0.01 --l2-adam-eps 0.00000001 --hidden-channels 32 --meta-sgd-mode clamp --meta-sgd-clamp-min 0.000001 --meta-sgd-clamp-max 0.2 --plots --output-root outputs_review_seed%%S
  if errorlevel 1 goto failed
)
python ..\tools\export_results.py --input-roots outputs_review_seed42 outputs_review_seed43 outputs_review_seed44 --output-dir ..\results\review_v2 --feedback-zip outputs_review_feedback_clamp.zip
if errorlevel 1 goto failed
echo All seeds completed. New clamp results are in ..\results\review_v2.
popd
pause
exit /b 0
:failed
echo Failed. No further seeds were launched. Copy the full error before closing.
popd
pause
exit /b 1
