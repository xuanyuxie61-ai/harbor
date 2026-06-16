# Reverse-Engineer Project 288: 边界等离子体输运与偏滤器热负荷模拟系统

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `288`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for 边界等离子体输运与偏滤器热负荷模拟系统 is:

```text
synthesis-python-288 1.0
```

Project 288 is framed as a cleanroom reproduction task around computational plasma physics. The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

## Observable behavior

Reconstruct the command-line behavior for this computational plasma physics target:

```text
边界等离子体输运与偏滤器热负荷模拟系统
```

Start by matching these lines:

```text
边界等离子体输运与偏滤器热负荷模拟系统
Edge Plasma Transport & Divertor Heat Flux Simulation
High-Order Finite Difference & Stability Analysis
(小规模可复现实验)
[阶段 1/8] 物理参数初始化与基本物理量计算
Coulomb对数 ln(Λ)      = 13.089
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 288 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
