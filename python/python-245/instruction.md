# Reverse-Engineer Project 245: Nuclear Fission Simulation

## Cleanroom objective

You are in `/app/workspace` with a reference executable for project `245`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Project 245's identity check should return:

```text
synthesis-python-245 1.0
```

The task is to rebuild a small Python implementation that behaves like a reference fusion and radiation transport executable. For this task, stdout formatting matters: section labels and selected numbers are part of the public contract.

## Behavioral target

The public-facing subject of project 245 is:

```text
Nuclear Fission Simulation
```

Visible cues for the fusion and radiation transport workflow:

```text
PROJECT_245: Nuclear Fission Simulation
Fragment Mass Distribution & Energy Release Modelling
High-Order Finite Differences & Stability Analysis
Binding energy per nucleon B/A (Liquid Drop Model):
1. Nuclear Constants and Binding Energy [nuclear_constants.py]
Nucleus        A     Z   B(A,Z) [MeV]    B/A [MeV]
```

The final workspace for project 245 needs your implementation and a build script that regenerates `./executable`.

Do not copy, unpack, wrap, or call the supplied binary while solving Nuclear Fission Simulation; create a fresh implementation. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
