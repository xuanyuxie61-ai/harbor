# Reverse-Engineer Project 207: 随机热传导方程的不确定性量化与置信/预测带构建

## Black-box target

You are in `/app/workspace` with a reference executable for project `207`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Use this version text to confirm you are probing the right oracle:

```text
synthesis-python-207 1.0
```

Project 207 asks for a faithful external clone of a synthesized uncertainty quantification report generator. The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

## Reference behavior

The binary's scientific topic is:

```text
随机热传导方程的不确定性量化与置信/预测带构建
```

The transcript begins to define the target through:

```text
随机热传导方程的不确定性量化与置信/预测带构建
Scientific Problem: UQ for Stochastic Parabolic PDEs
Focus: Confidence Intervals & Prediction Intervals
SHE-UQ 问题参数摘要
空间域:        [0, 1.0] m,  nx=51,  dx=0.020000
时间域:        [0, 0.5] s,  nt=101,  dt=0.005000
```

Your rebuilt uncertainty quantification program must include a build step, either `compile.sh` or `build.sh`, that creates `/app/workspace/executable`.

Do not use internet access or outside source recovery for 随机热传导方程的不确定性量化与置信/预测带构建; infer behavior from the local artifacts. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
