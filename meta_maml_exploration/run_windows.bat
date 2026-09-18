@echo off
chcp 65001 > nul
echo 正在运行 MAML、Meta-SGD、固定 L2 和 Meta-L2 四组实验...
python run_all.py --download --episodes 600 --eval-episodes 200 --meta-batch 4
if errorlevel 1 (
  echo.
  echo 运行失败，请复制上方报错信息进行检查。
) else (
  echo.
  echo 运行完成。请打开 outputs 文件夹查看结果。
)
pause
