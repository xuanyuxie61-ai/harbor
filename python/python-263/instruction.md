# Reverse-Engineer Project 263: Coronal Heating & Solar Wind Acceleration

## Cleanroom objective

You are in `/app/workspace` with a reference executable for project `263`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Project 263's identity check should return:

```text
synthesis-python-263 1.0
```

The benchmark centers on a scientific driver in numerical-methods benchmark, with behavior exposed through a local binary. A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

## Behavioral target

The public-facing subject of project 263 is:

```text
Coronal Heating & Solar Wind Acceleration
```

Visible cues for the numerical-methods benchmark workflow:

```text
PROJECT 263 - Coronal Heating & Solar Wind Acceleration
High-Order Finite Differences with Stability Analysis
Python 3.11.11, NumPy 1.26.4
Random seed: 263
Corona loop grid (nodes=64)
n_nodes                 : 64
```

The final workspace for project 263 needs your implementation and a build script that regenerates `./executable`.

Do not copy, unpack, wrap, or call the supplied binary while solving Coronal Heating & Solar Wind Acceleration; create a fresh implementation. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
