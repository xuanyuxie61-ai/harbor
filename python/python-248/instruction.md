# Reverse-Engineer Project 248: Galaxy Formation Hydrodynamics Test Problem

## Task target

You are in `/app/workspace` with a reference executable for project `248`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for Galaxy Formation Hydrodynamics Test Problem is:

```text
synthesis-python-248 1.0
```

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in astrophysics simulation. The binary is meant for observation only; rebuild the behavior in your own files after probing it.

## External contract

Use the following label as the center of the reconstruction:

```text
Galaxy Formation Hydrodynamics Test Problem
```

Reference lines worth preserving for project 248:

```text
PROJECT_248 :: Galaxy Formation Hydrodynamics Test Problem
Domain: high-order finite differences + stability analysis
(small-scale reproducible experiment)
N     dx           L_inf (order 4)   L_inf (order 6)   rate_4  rate_6
[ok] all 16 Python modules imported successfully
High-order FD convergence check: f(x) = sin(2 pi x) on [0, 1]
```

For project 248, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 248, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
