# Reverse-Engineer Project 214: L1-Regularized Sparse PCE Recovery

## Build goal

You are in `/app/workspace` with a reference executable for project `214`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 214:

```text
synthesis-python-214 1.0
```

The program under observation is a deterministic scientific computing demonstrator packaged as a single executable. Capture enough of the ordinary run to reproduce both the scientific storyline and the small numeric landmarks.

## What must match

Project 214 should be rebuilt around:

```text
L1-Regularized Sparse PCE Recovery
```

Useful observation anchors from the oracle:

```text
Project 214: L1-Regularized Sparse PCE Recovery
for Stochastic Elliptic PDEs
阶段 1: 稀疏多项式基构造
维数 d = 3,  总阶 p = 4
全阶截断基大小 M_full = 35
双曲交叉基大小 M_hc  = 11 (q=0.6)
```

Finish by making a standalone workspace executable for L1-Regularized Sparse PCE Recovery; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 214, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
