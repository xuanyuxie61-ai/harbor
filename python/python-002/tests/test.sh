#!/bin/bash
# tests/test.sh — Harbor 测试运行脚本
# 将 test_main.py 复制到 /app/workspace，运行后删除

set +e  # 允许测试失败，捕获退出码

echo "=========================================="
echo "  Harbor Test Runner"
echo "  Workspace: /app/workspace"
echo "  Tests:     /tests"
echo "=========================================="

# 确保日志目录存在
mkdir -p /logs/verifier

# 1. 复制 test_main.py 到 workspace
echo "[1/3] 复制 test_main.py → /app/workspace/"
cp /tests/test_main.py /app/workspace/

# 2. 在 workspace 中运行测试
echo "[2/3] 运行 test_main.py ..."
cd /app/workspace
python test_main.py
EXIT_CODE=$?

# 3. 删除 test_main.py
echo "[3/3] 清理 test_main.py"
rm -f /app/workspace/test_main.py

# 生成奖励文件
if [ $EXIT_CODE -eq 0 ]; then
    echo '{"score": 1.0, "passed": 30, "total": 30, "status": "pass", "failures": []}' > /logs/verifier/reward.json
    echo "score=1.0" > /logs/verifier/reward.txt
    echo "passed=30" >> /logs/verifier/reward.txt
    echo "total=30" >> /logs/verifier/reward.txt
    echo "status=pass" >> /logs/verifier/reward.txt
    echo ""
    echo "=========================================="
    echo "  全部 30 个测试通过!"
    echo "=========================================="
else
    echo "{\"score\": 0.0, \"passed\": 0, \"total\": 30, \"status\": \"fail\", \"failures\": [\"exit_code=$EXIT_CODE\"]}" > /logs/verifier/reward.json
    echo "score=0.0" > /logs/verifier/reward.txt
    echo "passed=0" >> /logs/verifier/reward.txt
    echo "total=30" >> /logs/verifier/reward.txt
    echo "status=fail" >> /logs/verifier/reward.txt
    echo ""
    echo "=========================================="
    echo "  测试失败 (exit=$EXIT_CODE)"
    echo "=========================================="
fi

echo "  奖励文件: /logs/verifier/reward.json"
exit $EXIT_CODE
