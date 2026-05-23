# Harbor Python Benchmark 项目说明

> 路径: `/mnt/data/zpy/sci-swe/source code/harbor/python/`

---

## 项目概览

| 项目 | 科学主题 | 测试用例数 | 难度 |
|------|----------|-----------|------|
| python-001 | 不规则小行星多尺度引力场建模 | 28 | hard |
| python-002 | 恒星演化与核合成多物理耦合模拟 | 30 | hard |

---

## 目录结构（通用）

```
python-XXX/
├── Dockerfile
├── task.toml
├── instruction.md
├── solution/                         # 空，预留给答案
├── environment/
│   └── workspace/                    # 被 COPY 到容器 /app/workspace
│       ├── main.py                   # 主入口（含完整 demo 流程）
│       ├── module_*.py               # 科学计算模块
│       └── ...
└── tests/                            # 被 COPY 到容器 /tests
    ├── test.sh                       # 测试运行脚本
    └── test_main.py                  # 测试用例文件
```

---

## python-001 vs python-002 核心区别

### 1. test_main.py 的运行方式

**两个项目现在统一使用相同的运行方式**：

- test.sh 通过 `PYTHONPATH=/app/workspace python /tests/test_main.py` 设置模块搜索路径
- test_main.py 始终留在 `/tests/` 目录，不复制到 workspace，不污染源码
- 导入方式：项目内模块通过 `from module_xxx import ...` 相对导入，PYTHONPATH 让 Python 能找到它们

唯一区别：python-001 的 test_main.py 额外加了 `sys.path.insert(0, '/app/workspace')` 作为双保险。

### 2. test.sh 流程（两项目相同）

---

## test_main.py 结构

两个项目的 test_main.py 结构相同：

```python
# 1. 模块导入
import numpy as np
from module_xxx import ...

# 2. 主函数 main()（完整的科学计算 demo）
def main():
    ...

if __name__ == "__main__":
    main()

# 3. 测试用例（inline，紧跟 main 之后执行）
# ---- TC01: 描述 ----
result = function_call(...)
assert condition, '[TC01] 描述 FAILED'

# ... 更多测试用例 ...

print('\n全部 N 个测试通过!\n')
```

测试用例在 `main()` 执行完毕后依次运行，共享 main() 中创建的变量和对象。

---

## 奖励输出规范

两个项目统一输出到 `/logs/verifier/`：

| 文件 | 格式 | 内容 |
|------|------|------|
| `reward.json` | JSON | `{"score": 1.0, "passed": N, "total": N, "status": "pass", "failures": []}` |
| `reward.txt` | 纯文本 | `score=1.0` / `passed=N` / `total=N` / `status=pass` |

- 全部测试通过 → `score=1.0`, `status="pass"`
- 任一测试失败 → `score=0.0`, `status="fail"`
- 运行异常（exit code ≠ 0）→ `score=0.0`, `status="fail"`

---

## Dockerfile 差异

```dockerfile
# python-001: test_main.py 从 /tests 运行
COPY workspace/. /app/workspace
COPY tests/. /tests

# python-002: test_main.py 需要被复制到 workspace（test.sh 中处理）
COPY workspace/. /app/workspace
COPY tests/. /tests
```

两者 Dockerfile 相同：都 COPY workspace 和 tests 到容器。区别在 test.sh 的行为。

---

## 选择指南

| 场景 | 说明 |
|------|------|
| 通用 | 两个项目现统一使用 PYTHONPATH 模式，无需选择 |
| python-001 | test_main.py 有 `sys.path.insert` 双保险 |
| python-002 | 纯依赖 PYTHONPATH，test.sh 最简 |
