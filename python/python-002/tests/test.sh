#!/bin/bash
# tests/test.sh — Harbor 测试运行脚本
# 通过 PYTHONPATH 让 /tests/test_main.py 可以导入 /app/workspace 中的模块
# 不再复制 test_main.py，直接从 /tests 目录运行

echo "=========================================="
echo "  Harbor Test Runner"
echo "  Test dir:  /tests"
echo "  Workspace: /app/workspace"
echo "=========================================="

# 确保日志目录存在
mkdir -p /logs/verifier

# 运行测试：
#   PYTHONPATH=/app/workspace → 让 Python 能找到 workspace 中的模块
PYTHONPATH=/app/workspace python /tests/test_main.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo '{"score":1.0,"passed":30,"total":30,"status":"pass","failures":[]}' > /logs/verifier/reward.json
  echo "1.0" > /logs/verifier/reward.txt
else
  echo '{"score":0.0,"passed":0,"total":30,"status":"fail","failures":["test_failed"]}' > /logs/verifier/reward.json
  echo "0.0" > /logs/verifier/reward.txt
fi

echo ""
echo "=========================================="
echo "  测试完成，奖励文件位于 /logs/verifier/"
echo "=========================================="
