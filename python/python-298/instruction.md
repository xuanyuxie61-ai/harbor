# Reverse-Engineer Project 298: 高阶有限差分 PIC 稳定性分析

## Build goal

You are in `/app/workspace` with a reference executable for project `298`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 298:

```text
synthesis-python-298 1.0
```

The visible binary represents a synthesized computational plasma physics workflow with a deterministic command-line transcript. Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

## What must match

Project 298 should be rebuilt around:

```text
高阶有限差分 PIC 稳定性分析
```

Useful observation anchors from the oracle:

```text
PROJECT 298: 高阶有限差分 PIC 稳定性分析
电子数密度 n_e        = 1.000e+20 m^-3
电子温度 T_e          = 100.00 eV (1.160e+06 K)
等离子体频率 omega_pe = 5.641e+11 rad/s
[PART 1] 等离子体物理参数
德拜长度 lambda_D     = 7.434e-06 m
```

Finish by making a standalone workspace executable for 高阶有限差分 PIC 稳定性分析; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 298, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
