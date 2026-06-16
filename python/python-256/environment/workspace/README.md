# 计算天体物理——恒星振动模式与星震学反演

The task is to rebuild a small Python implementation that behaves like a reference astrophysics simulation executable.

The program reads like a research demonstration: it sets up model parameters, evolution loops, and summary tables, then prints a staged report with deterministic diagnostics. A robust answer separates command dispatch from numeric helpers and final text rendering.

## What to reproduce

- PROJECT_256 统一入口: 计算天体物理——恒星振动模式与星震学反演
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 本程序执行完整的星震学计算流程:
- 1. 构建恒星平衡模型 (多方球)

## Command surface

```bash
./executable --help
./executable --version
./executable
```

For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

Expected `--version` text:

```text
synthesis-python-256 1.0
```

## Output cues

```text
PROJECT_256: 计算天体物理——恒星振动模式与星震学反演
高阶有限差分与稳定性分析 (小规模可复现实验)
阶段 1: 恒星平衡模型构建
恒星质量: 1.0000 M_sun
恒星半径: 1.0000 R_sun
多方指数: n = 3.0
平均分子量: μ = 0.6173
中心密度: ρ_c = 1.1078e+01 g/cm³
中心压强: P_c = 5.2846e+15 dyn/cm²
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Preserve non-English labels and punctuation where they appear in the reference transcript.
