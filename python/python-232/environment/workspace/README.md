# 计算高能物理散射振幅数值计算与截面积分

The visible binary represents a synthesized lattice field theory workflow with a deterministic command-line transcript.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. The hidden tests may look beyond the examples, so preserve the general report shape rather than only one line.

## What to reproduce

- main.py — PROJECT_232 统一入口 (PROJECT_232)
- 计算高能物理: 散射振幅数值计算与截面积分
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 本文件是整个合成项目的统一入口。零参数可直接运行。

## Command surface

```bash
./executable --help
./executable --version
./executable
```

Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

Expected `--version` text:

```text
synthesis-python-232 1.0
```

## Output cues

```text
PROJECT_232: 计算高能物理散射振幅数值计算与截面积分
高阶有限差分与稳定性分析 (小规模可复现实验)
物理过程: ππ → ππ 弹性散射 (I=2 通道)
粒子质量: m_π± = 0.139570 GeV, m_π⁰ = 0.134977 GeV
阈值: √s_thr = 2m_π = 0.279141 GeV
1. 运动学网格构造
[1a] 线性网格: √s ∈ [0.2891, 0.5000] GeV
网格点数: 50, 间距 Δ√s = 0.004303 GeV
[1b] 切比雪夫网格: 50 点
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.
