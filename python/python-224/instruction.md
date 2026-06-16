# Reverse-Engineer Project 224: Higgs Decay Signal Strength Fitting

## Task target

You are in `/app/workspace` with a reference executable for project `224`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for Higgs Decay Signal Strength Fitting is:

```text
synthesis-python-224 1.0
```

This case exposes the observable surface of a research-style numerical-methods benchmark driver rather than its source. Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

## External contract

Use the following label as the center of the reconstruction:

```text
Higgs Decay Signal Strength Fitting
```

Reference lines worth preserving for project 224:

```text
PROJECT_224 :: Higgs Decay Signal Strength Fitting
High-Order Finite Differences & Stability Analysis
Computational High-Energy Physics (small-scale reproducible experiment)
sin^2 theta_W = 0.222897
Recovered VEV          : v = 230.900734 GeV  (input 246.2196)
V''(v)                 : 16116.0408 GeV^2   (m_h^2 = 15647.51)
```

For project 224, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 224, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
