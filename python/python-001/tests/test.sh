#!/bin/bash
# tests/test.sh — Harbor 测试运行脚本
# 从 /tests/ 目录运行 python /tests/test_main.py
# 测试 /app/workspace/ 中的科学计算代码

set -e

echo "=========================================="
echo "  Harbor Test Runner"
echo "  Test dir:  /tests"
echo "  Workspace: /app/workspace"
echo "=========================================="

# 确保日志目录存在
mkdir -p /logs/verifier

# 运行测试
python /tests/test_main.py

echo ""
echo "=========================================="
echo "  测试完成，奖励文件位于 /logs/verifier/"
echo "=========================================="
