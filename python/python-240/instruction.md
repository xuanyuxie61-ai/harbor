# Reverse-Engineer Project 240: 重离子碰撞椭圆流与初始条件涨落建模

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `240`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for 重离子碰撞椭圆流与初始条件涨落建模 is:

```text
synthesis-python-240 1.0
```

Project 240 asks for a faithful external clone of a synthesized uncertainty quantification report generator. The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

## Observable behavior

Reconstruct the command-line behavior for this uncertainty quantification target:

```text
重离子碰撞椭圆流与初始条件涨落建模
```

Start by matching these lines:

```text
重离子碰撞椭圆流与初始条件涨落建模
Heavy-Ion Collision: Elliptic Flow & Initial Condition Fluctuations
High-Order Finite Differences & Stability Analysis
1. 蒙特卡洛Glauber初始条件 (Monte Carlo Glauber)
碰撞系统: Au+Au, √s = 200 GeV
碰撞参数: b = 7.0 fm
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 240 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
