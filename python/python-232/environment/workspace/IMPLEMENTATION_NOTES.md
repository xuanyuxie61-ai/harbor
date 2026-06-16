# Reconstruction Notes: Project 232

The cleanroom notes below emphasize small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: s_matrix
- Diagnostics/reporting: cross_section, special_functions, spectral_analysis, stability
- Supporting modules: constants, finite_difference, kinematics, lattice_utils, partial_wave, phase_shift

## Suggested coding path

- Keep project 232's build script idempotent so repeated verifier runs do not change behavior.
- Match the order and magnitude of 计算高能物理散射振幅数值计算与截面积分's reported values before adding extra scientific machinery.
- Mirror the visible cadence of the lattice field theory transcript: setup, computation, diagnostics, summary.
- If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `计算高能物理散射振幅数值计算与截面积分`; hidden tests are not limited to them.

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
权重和: 2.000329
```

Keep constants close to the observed transcript, but avoid copying source files from outside the workspace. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
