@echo off
chcp 65001 > nul
echo 进行四种方法的小规模流程检查，每种方法只训练 3 个 episode。
python run_all.py --download --episodes 3 --eval-episodes 3 --meta-batch 1 --output-root outputs_quick_check
pause
