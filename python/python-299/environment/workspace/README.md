# Fokker-Planck 碰撞输运计算

Project 299 is framed as a cleanroom reproduction task around uncertainty quantification.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. Use fixed seeds and stable constants where the reference advertises reproducibility.

## Workflow outline

- 博士级科研代码合成说明
- 项目名称
- Fokker-Planck 碰撞输运计算: 高阶有限差分与稳定性分析
- 科学问题

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

Version output:

```text
synthesis-python-299 1.0
```

## Stable landmarks

```text
Fokker-Planck 碰撞输运计算
高阶有限差分与稳定性分析 — 博士级科学计算项目
儒略日: 2461207
Unix 时间戳: 1781543210.293
§1 等离子体参数设置
等离子体参数摘要
ln(Lambda)    = 16.3040
碰撞时间 τ_c = 4.3011e+15 s
碰撞频率 ν_c = 2.3250e-16 s⁻¹
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.
