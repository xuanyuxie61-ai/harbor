# 空间等离子体波粒相互作用

The reference executable is the oracle for a reduced computational plasma physics experiment with stable printed diagnostics.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. Do not attempt to preserve the original package structure unless it helps; match behavior before architecture.

## Workflow outline

- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 物理模型: 1D 静电 Vlasov-Poisson 系统
- 数值方法: 半拉格朗日 + 谱方法 Poisson 求解器
- 分析工具: von Neumann 稳定性 / Hankel 模态分解

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

Version output:

```text
synthesis-python-293 1.0
```

## Stable landmarks

```text
空间等离子体波粒相互作用
高阶有限差分与稳定性分析 (小规模可复现实验)
物理模型: 1D 静电 Vlasov-Poisson 系统
数值方法: 半拉格朗日 + 谱方法 Poisson 求解器
分析工具: von Neumann 稳定性 / Hankel 模态分解
实验 1: Landau 阻尼
网格: nx=32, nv=64
空间: [0.00, 12.57], dx=0.3927
速度: [-6.00, 6.00], dv=0.1875
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Probe before coding, because several projects share generic themes but differ in their visible numbers.
