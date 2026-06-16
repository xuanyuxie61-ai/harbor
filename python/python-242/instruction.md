# Reverse-Engineer Project 242: Nuclear shell-model energy levels and transition

## Task target

You are in `/app/workspace` with a reference executable for project `242`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The stable identity string for Nuclear shell-model energy levels and transition is:

```text
synthesis-python-242 1.0
```

This benchmark instance is a black-box reconstruction exercise for a compact fusion and radiation transport program. Treat the default execution as the primary specification and preserve the order of its visible sections.

## External contract

Use the following label as the center of the reconstruction:

```text
Nuclear shell-model energy levels and transition
```

Reference lines worth preserving for project 242:

```text
PROJECT 242: Nuclear shell-model energy levels and transition
probabilities with high-order finite differences and stability
analysis (small-scale, reproducible experiment)
Step 0: Problem specification
Nucleus         : 18O (Z=8, N=10)
Core            : 16O (Z=8, N=8)
```

For project 242, the deliverable is original Python plus a build script that produces `executable` at the workspace root.

Hidden tests may remove the visible binary for project 242, so your build must stand alone. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
