# Reverse-Engineer Project 294: 激光等离子体相互作用: 高阶有限差分与稳定性分析

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `294`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for 激光等离子体相互作用: 高阶有限差分与稳定性分析 is:

```text
synthesis-python-294 1.0
```

This is a ProgramBench-style task: infer and reproduce a uncertainty quantification CLI from documentation plus black-box runs. Start with the version and help flags, then run the binary without arguments to collect the full staged report.

## Observable behavior

Reconstruct the command-line behavior for this uncertainty quantification target:

```text
激光等离子体相互作用: 高阶有限差分与稳定性分析
```

Start by matching these lines:

```text
激光等离子体相互作用: 高阶有限差分与稳定性分析
High-Order Finite Difference for Laser-Plasma Interaction
配置创建完成: N_x=256, N_v=128, FD_order=4
阶段 1: 物理参数设置与完整性校验
参考密度 n_0 = 1.00e+25 m^-3
电子温度 T_e = 1.60e-16 J (1000 eV)
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 294 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
