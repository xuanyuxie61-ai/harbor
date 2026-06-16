# Reverse-Engineer Project 293: 空间等离子体波粒相互作用

## Cleanroom objective

You are in `/app/workspace` with a reference executable for project `293`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Project 293's identity check should return:

```text
synthesis-python-293 1.0
```

The reference executable is the oracle for a reduced computational plasma physics experiment with stable printed diagnostics. The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

## Behavioral target

The public-facing subject of project 293 is:

```text
空间等离子体波粒相互作用
```

Visible cues for the computational plasma physics workflow:

```text
空间等离子体波粒相互作用
高阶有限差分与稳定性分析 (小规模可复现实验)
物理模型: 1D 静电 Vlasov-Poisson 系统
数值方法: 半拉格朗日 + 谱方法 Poisson 求解器
分析工具: von Neumann 稳定性 / Hankel 模态分解
实验 1: Landau 阻尼
```

The final workspace for project 293 needs your implementation and a build script that regenerates `./executable`.

Do not copy, unpack, wrap, or call the supplied binary while solving 空间等离子体波粒相互作用; create a fresh implementation. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
