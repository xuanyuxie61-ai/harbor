# 中子星核物质状态方程约束

Project 244 is framed as a cleanroom reproduction task around uncertainty quantification.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. Use fixed seeds and stable constants where the reference advertises reproducibility.

## What to reproduce

- 中子星核物质状态方程约束：高阶有限差分与稳定性分析
- 博士级合成说明文档
- 科学领域**：中子星物理 / 核天体物理 / 计算天体物理
- 计算难度**：博士级前沿科学计算

## Command surface

```bash
./executable --help
./executable --version
./executable
```

The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

Expected `--version` text:

```text
synthesis-python-244 1.0
```

## Output cues

```text
中子星核物质状态方程约束
高阶有限差分与稳定性分析 (小规模可复现实验)
Python: 3.11.11
NumPy: 1.26.4
阶段 1: 数值基础验证
机器精度 eps = 2.220446e-16
Gamma(5) = 24.0000000000  (精确 24)
ln Gamma(10) = 12.8018274801
Legendre P_5 零点: [-0.90617985 -0.53846931  0.          0.53846931  0.90617985]
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.
