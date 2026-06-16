# Reconstruction Notes: Project 285

This task can be solved by rebuilding a concise pipeline for magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Practical module split

- Numerical layer: banded_solver, high_order_fd, mesh_domain_topology, time_integrator
- Diagnostics/reporting: md_trajectory_analysis, order_parameter_analysis, simulation_engine
- Supporting modules: brillouin_zone, entropy_trace, lgd_free_energy, magnetoelectric_bounds, magnetoelectric_connector, multiferroic_constants

## Solver-shaped plan

- For this computational plasma physics task, treat stdout as an API and keep flag output separate from the scientific transcript.
- Model only the numerical detail that supports project 285's printed diagnostics.
- Keep the final report for project 285 deterministic, compact, and ordered like the reference.
- For 多铁性材料磁电耦合模拟, do not include the original executable in the rebuilt solution or call it from a wrapper.

## Transcript details

Use these public lines as reconstruction checkpoints for `多铁性材料磁电耦合模拟`; hidden tests are not limited to them.

```text
PROJECT 285: 多铁性材料磁电耦合模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
材料体系: BiFeO₃ (BFO) 钙钛矿多铁性材料
理论框架: Landau-Ginzburg-Devonshire 自由能泛函
数值方法: 6阶有限差分 + 半隐式时间积分 + ADI 分解
1. 数值方法验证
[1.1] 高阶有限差分精度测试
一阶导数误差:
O(h²)  : 4.0246e-02
O(h⁴)  : 1.9434e-05
```

Make the no-argument path finish quickly under verifier time limits. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
