# 计算宇宙学: 再电离历史数值模拟

This is a ProgramBench-style task: infer and reproduce a astrophysics simulation CLI from documentation plus black-box runs.

The program reads like a research demonstration: it sets up model parameters, evolution loops, and summary tables, then prints a staged report with deterministic diagnostics. The verifier rewards the public interface: executable creation, flags, deterministic output, and selected anchors.

## Visible purpose

- 高阶有限差分与 IMEX 稳定性分析 (小规模可复现实验)
- 项目结构:
- reion_cosmology.py       : 物理与宇宙学常数
- reion_grid.py            : 计算网格生成

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

Start with the version and help flags, then run the binary without arguments to collect the full staged report.

Reference version:

```text
synthesis-python-261 1.0
```

## Reference cues

```text
#  计算宇宙学: 再电离历史数值模拟
#  高阶有限差分与 IMEX 稳定性分析 (小规模可复现实验)
reion_cosmology.py       : 物理与宇宙学常数
reion_grid.py            : 计算网格生成
计算宇宙学: 再电离历史数值模拟
高阶有限差分与 IMEX 稳定性分析 (小规模可复现实验)
项目结构:
reion_transfer.py        : 辐射传输 BVP
reion_recombination.py   : 复合动力学 (含延迟反馈)
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep the generated executable at the workspace root and make it executable.
