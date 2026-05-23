#!/bin/bash
# tests/test.sh — Harbor 测试运行脚本
# 通过 PYTHONPATH 让 /tests/test_main.py 可以导入 /app/workspace 中的模块

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
#   test_main.py 内部也有 sys.path.insert(0, '/app/workspace') 作为双保险
PYTHONPATH=/app/workspace python /tests/test_main.py

echo ""
echo "=========================================="
echo "  测试完成，奖励文件位于 /logs/verifier/"
echo "=========================================="
