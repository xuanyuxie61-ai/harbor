#!/bin/bash
# tests/test.sh — Harbor 测试运行脚本
# 通过 PYTHONPATH 让 /tests/test_main.py 可以导入 /app/workspace 中的模块
# 不再复制 test_main.py，直接从 /tests 目录运行

set -e

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

echo ""
echo "=========================================="
echo "  测试完成，奖励文件位于 /logs/verifier/"
echo "=========================================="
