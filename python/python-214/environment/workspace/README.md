# L1-Regularized Sparse PCE Recovery

The program under observation is a deterministic scientific computing demonstrator packaged as a single executable.

The program reads like a research demonstration: it sets up configuration values, numerical kernels, and formatted report sections, then prints a staged report with deterministic diagnostics. Exact internal algorithms are less important than externally stable scientific summaries and return codes.

## What to reproduce

- main.py — 稀疏 PCE 恢复博士级计算项目主入口
- 项目主题:
- L1 正则化稀疏多项式混沌展开恢复
- ——面向随机椭圆 PDE 的不确定性量化

## Command surface

```bash
./executable --help
./executable --version
./executable
```

Capture enough of the ordinary run to reproduce both the scientific storyline and the small numeric landmarks.

Expected `--version` text:

```text
synthesis-python-214 1.0
```

## Output cues

```text
Project 214: L1-Regularized Sparse PCE Recovery
for Stochastic Elliptic PDEs
阶段 1: 稀疏多项式基构造
维数 d = 3,  总阶 p = 4
全阶截断基大小 M_full = 35
双曲交叉基大小 M_hc  = 11 (q=0.6)
Legendre 正交性验证误差 (16 对): 8.218e-16
8 点 Gauss-Jacobi (α=β=0.5) 精确阶: 15 (理论 = 15)
阶段 2: 观测算子与采样
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.
