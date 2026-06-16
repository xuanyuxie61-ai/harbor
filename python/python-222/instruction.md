# Reverse-Engineer Project 222: PartonShowerHD v1.0

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `222`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for PartonShowerHD v1.0 is:

```text
synthesis-python-222 1.0
```

Project 222 is framed as a cleanroom reproduction task around numerical-methods benchmark. The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

## Observable behavior

Reconstruct the command-line behavior for this numerical-methods benchmark target:

```text
PartonShowerHD v1.0
```

Start by matching these lines:

```text
PartonShowerHD v1.0
高阶有限差分稳定性分析下的 Parton Shower 与强子化
计算高能物理 · 博士级科学计算合成项目
Stage 1: 物理参数初始化
beta_0(nf=5) = 0.610094
跑动耦合常数 alpha_s(mu):
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 222 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
