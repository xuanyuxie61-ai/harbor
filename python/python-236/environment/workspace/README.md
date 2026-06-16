# 格点 QCD: 强子谱关联函数拟合

The program under observation is a deterministic lattice field theory demonstrator packaged as a single executable.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. Exact internal algorithms are less important than externally stable scientific summaries and return codes.

## Scientific role

- main.py — 格点 QCD 强子谱关联函数拟合: 高阶有限差分与稳定性分析
- 主入口文件, 零参数可运行.
- 完整流程:
- 1. 构建格点几何

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

Capture enough of the ordinary run to reproduce both the scientific storyline and the small numeric landmarks.

Stable version text:

```text
synthesis-python-236 1.0
```

## Report signatures

```text
格点 QCD: 强子谱关联函数拟合
高阶有限差分与稳定性分析 (小规模可复现实验)
1. 格点几何构建
格点: Ls=4, Lt=12, V=768
形状: (4, 4, 4, 12)
距离矩阵: max=6.928, mean=3.825
半径 1.5 内邻居数: 33
连通性已保存到: lattice_io.json
2. 规范场与 Wilson 梯度流
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.
