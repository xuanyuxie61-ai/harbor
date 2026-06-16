# 设计一维催化反应器的密度分布 ρ(x) ∈ [0,1]^n,

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in scientific computing.

The program reads like a research demonstration: it sets up configuration values, numerical kernels, and formatted report sections, then prints a staged report with deterministic diagnostics. A compact implementation is acceptable when it keeps the same CLI, exit status, section order, and recognizable diagnostics.

## Workflow outline

- 催化反应器多目标拓扑优化 — 博士级科学计算合成项目
- PROJECT_215 合成说明
- 应用问题**: 一维催化反应器的功能梯度材料拓扑优化
- 难度级别**: 博士级前沿科学计算

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

The binary is meant for observation only; rebuild the behavior in your own files after probing it.

Version output:

```text
synthesis-python-215 1.0
```

## Stable landmarks

```text
f1: 结构柔度    (最小化 → 最大化刚度)
f2: 热应力方差  (最小化 → 均匀温度分布)
f3: 负转化率    (最小化 → 最大化生化转化效率)
约束: 体积分数 ≤ 0.5
科学问题:
设计一维催化反应器的密度分布 ρ(x) ∈ [0,1]^n,
同时优化:
方法: NSGA-II + 顺序耦合 PDE + 随机鲁棒性
[1.1] 数值积分精度验证:
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Be careful with warning text: hidden tests generally inspect stdout and exit behavior.
