# Reverse-Engineer Project 258: 弱引力透镜质量重建: 高阶有限差分与稳定性分析

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `258`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for 弱引力透镜质量重建: 高阶有限差分与稳定性分析 is:

```text
synthesis-python-258 1.0
```

The program under observation is a deterministic astrophysics simulation demonstrator packaged as a single executable. Capture enough of the ordinary run to reproduce both the scientific storyline and the small numeric landmarks.

## Observable behavior

Reconstruct the command-line behavior for this astrophysics simulation target:

```text
弱引力透镜质量重建: 高阶有限差分与稳定性分析
```

Start by matching these lines:

```text
弱引力透镜质量重建: 高阶有限差分与稳定性分析
Weak Lensing Mass Reconstruction with High-Order FD
PROJECT 258 — 计算宇宙学博士级可复现实验
阶段 1: 宇宙学距离与临界密度计算
透镜红移 z_L = 0.3
源红移   z_S = 1.0
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 258 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
