# 激光等离子体相互作用: 高阶有限差分与稳定性分析

This is a ProgramBench-style task: infer and reproduce a uncertainty quantification CLI from documentation plus black-box runs.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. The verifier rewards the public interface: executable creation, flags, deterministic output, and selected anchors.

## Observed behavior

- main.py - 激光等离子体相互作用高阶有限差分仿真统一入口
- 计算等离子体: 激光等离子体相互作用
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- 本程序实现 1D1V Vlasov-Maxwell 方程组的高阶有限差分数值求解,

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

Start with the version and help flags, then run the binary without arguments to collect the full staged report.

Identity check:

```text
synthesis-python-294 1.0
```

## Transcript anchors

```text
激光等离子体相互作用: 高阶有限差分与稳定性分析
High-Order Finite Difference for Laser-Plasma Interaction
配置创建完成: N_x=256, N_v=128, FD_order=4
阶段 1: 物理参数设置与完整性校验
参考密度 n_0 = 1.00e+25 m^-3
电子温度 T_e = 1.60e-16 J (1000 eV)
等离子体频率 omega_p0 = 1.78e+14 rad/s
趋肤深度 c/omega_p0 = 1.68e-06 m
热速度 v_th = 1.33e+07 m/s (v_th/c = 0.0442)
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep the generated executable at the workspace root and make it executable.
