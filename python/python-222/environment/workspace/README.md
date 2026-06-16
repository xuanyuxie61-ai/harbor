# PartonShowerHD v1.0

Project 222 is framed as a cleanroom reproduction task around numerical-methods benchmark.

The program reads like a research demonstration: it sets up stencil coefficients, amplification factors, and convergence indicators, then prints a staged report with deterministic diagnostics. Use fixed seeds and stable constants where the reference advertises reproducibility.

## Observed behavior

- PartonShowerHD: 高阶有限差分稳定性分析下的 Parton Shower 与强子化模型
- > **计算高能物理博士级科学计算合成项目
- > 15 个种子项目 → 1 个完整的高能物理 Parton Shower 数值模拟平台
- 1. 科学问题定义

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

Identity check:

```text
synthesis-python-222 1.0
```

## Transcript anchors

```text
PartonShowerHD v1.0
高阶有限差分稳定性分析下的 Parton Shower 与强子化
计算高能物理 · 博士级科学计算合成项目
Stage 1: 物理参数初始化
beta_0(nf=5) = 0.610094
跑动耦合常数 alpha_s(mu):
Stage 2: DGLAP 分裂核多项式投影
Pqq(z) ~ -14.7937 + 536.444*z + -4403.56*z^2 + 13658.7*z^3 + -17661.3*z^4 + 8066.67*z^5
Pqg(z) ~ 0.5 + -1*z + 1*z^2 + -1.55188e-13*z^3 + 1.22402e-13*z^4 + -3.36606e-14*z^5
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.
