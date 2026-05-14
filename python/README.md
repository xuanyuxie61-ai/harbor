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

| | python-001 | python-002 |
|------|-----------|-----------|
| **运行位置** | 直接在 `/tests/` 下运行 | 先**复制到** `/app/workspace/`，再运行 |
| **运行命令** | `python /tests/test_main.py` | `cd /app/workspace && python test_main.py` |
| **运行后** | 文件保留在 `/tests/` | **删除** `/app/workspace/test_main.py` |
| **导入方式** | `sys.path.insert(0, '/app/workspace')` + 绝对路径导入 | `from stellar_grid import ...` 相对导入 |

### 2. 为什么不同？

**python-001（直接运行模式）**：
- `test_main.py` 始终留在 `/tests/` 目录，不污染 workspace
- 通过 `sys.path.insert(0, '/app/workspace')` 让 Python 能找到 workspace 中的模块
- 适合需要**严格隔离**测试代码和被测代码的场景
- test.sh 流程：**运行测试 → 生成奖励文件**（一步到位）

**python-002（复制运行模式）**：
- `test_main.py` 内部使用**相对导入**（`from stellar_grid import ...`），必须和模块在同一目录才能跑
- 复制到 workspace 后运行，测试完立刻删除，测试期间短暂共存
- 适合 test_main.py 本身就是 main.py 副本（含完整业务流程），需要在模块环境中运行
- test.sh 流程：**复制 → 运行 → 删除 → 生成奖励文件**（三步）

### 3. test.sh 流程对比

```
python-001:                          python-002:
┌──────────────────────┐             ┌──────────────────────┐
│ mkdir /logs/verifier │             │ mkdir /logs/verifier │
│ python /tests/test   │             │ cp tests → workspace │
│       _main.py       │             │ cd workspace         │
│ write reward.json    │             │ python test_main.py  │
└──────────────────────┘             │ rm test_main.py      │
                                     │ write reward.json    │
                                     └──────────────────────┘
```

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

| 场景 | 推荐模式 | 项目 |
|------|---------|------|
| test_main.py 仅含测试用例 | 直接运行 | python-001 |
| test_main.py = main.py + 测试用例（需在同一目录运行模块化代码） | 复制运行 | python-002 |
| 严格隔离测试和源码 | 直接运行 | python-001 |
| 测试需要访问 main.py 创建的运行时对象 | 复制运行 | python-002 |
