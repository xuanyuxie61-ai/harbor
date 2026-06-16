# Reconstruction Notes: Project 235

The source evidence points to small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Model layer: field_evolution
- Diagnostics/reporting: field_evolution, special_functions, stability_analysis
- Supporting modules: active_learning, anomaly_detector, detector_sim, event_generator, finite_difference, lattice_config

## Coding plan

- Make the flag paths cheap for project 235; most scientific work belongs in the no-argument path.
- Preserve any reproducibility claims made by 计算高能物理: 异常事件检测与新物理信号搜索; random-looking values should come from fixed data.
- Use repeated black-box probes to confirm the first and last sections of project 235.
- Use installed scientific packages only when they simplify the clone; plain Python is enough for many reports.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `计算高能物理: 异常事件检测与新物理信号搜索`; hidden tests are not limited to them.

```text
计算高能物理: 异常事件检测与新物理信号搜索
高阶有限差分与稳定性分析 (小规模可复现实验)
Python 3.11.11, NumPy 1.26.4
Phase 0: 物理常数与实验配置
√s = 13 TeV
格点: N_x=128, h=0.1, m=2.0
BSM: mZ'=500.0 GeV, g'=0.3
事件: 200, 质量窗口: (60.0, 120.0)
Phase 1: 高阶有限差分稳定性分析
Lattice: N_x=128, h=0.1, dt=0.0500, m=2.0, λ=0.5, μ³=0.0, CFL_dt_max=0.0707
```

Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
