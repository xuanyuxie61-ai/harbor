# 黑洞喷流形成与 MHD 模拟

The benchmark centers on a scientific driver in computational plasma physics, with behavior exposed through a local binary.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. The target is not source recovery. It is an original implementation that reproduces the black-box contract.

## Observed behavior

- PROJECT 252 — 黑洞喷流形成与 MHD 模拟：高阶有限差分与稳定性分析
- 博士级科研代码合成说明文档
- 一、项目概述
- 核心科学问题**：

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

Identity check:

```text
synthesis-python-252 1.0
```

## Transcript anchors

```text
PROJECT 252: 黑洞喷流形成与 MHD 模拟
高阶有限差分与稳定性分析 (博士级可复现实验)
黑洞喷流 MHD 模拟 & 稳定性分析 (博士级可复现实验)
完成 20 步, 最终密度扰动: 7.277096e-02
[1/8] 配置校验: 全部通过 (9 项)
[2/8] 网格生成: Nr=48, Nt=24, r=[2.00, 50.0]
[3/8] 初始条件: Bondi 吸积 + 磁化 + MRI 微扰
[4/8] 时间推进: dt=0.5000, n_steps=20, FD阶数=6
[5/8] 稳定性分析: 拟合增长率 gamma=-0.016985, R^2=0.9936
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make the no-argument path finish quickly under verifier time limits.
