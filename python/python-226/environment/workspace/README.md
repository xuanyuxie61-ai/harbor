# 中微子振荡概率与参数反演：高阶有限差分与稳定性分析

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in uncertainty quantification.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. A compact implementation is acceptable when it keeps the same CLI, exit status, section order, and recognizable diagnostics.

## What to reproduce

- 统一入口程序
- 科学问题：
- 在计算高能物理中，中微子振荡是探测基本粒子物理参数的重要工具。
- 本项目研究三代中微子在真空和物质中的传播，使用高阶有限差分方法

## Command surface

```bash
./executable --help
./executable --version
./executable
```

The binary is meant for observation only; rebuild the behavior in your own files after probing it.

Expected `--version` text:

```text
synthesis-python-226 1.0
```

## Output cues

```text
中微子振荡概率与参数反演：高阶有限差分与稳定性分析
博士级科研代码合成项目
本项目融合15个种子项目的核心算法，解决计算高能物理中的
中微子振荡参数反演问题。包含：
• 三代中微子振荡与MSW物质效应
• 高阶有限差分与von Neumann稳定性分析
• FEM太阳密度分布求解
• 带状矩阵存储与Jacobi迭代
• 矩阵指数精确演化
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Be careful with warning text: hidden tests generally inspect stdout and exit behavior.
