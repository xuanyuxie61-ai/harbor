# Reverse-Engineer Project 286: Tokamak Grad-Shafranov Equilibrium

## Build goal

You are in `/app/workspace` with a reference executable for project `286`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 286:

```text
synthesis-python-286 1.0
```

This benchmark instance is a black-box reconstruction exercise for a compact computational plasma physics program. Treat the default execution as the primary specification and preserve the order of its visible sections.

## What must match

Project 286 should be rebuilt around:

```text
Tokamak Grad-Shafranov Equilibrium
```

Useful observation anchors from the oracle:

```text
PROJECT_286: Tokamak Grad-Shafranov Equilibrium
High-order finite differences + MHD stability analysis
(small-scale reproducible experiment)
1. Geometry and plasma profiles
κ = 1.7,  δ = 0.33
grid: Nr=41, Nz=41,  dR=0.0275, dZ=0.0300
```

Finish by making a standalone workspace executable for Tokamak Grad-Shafranov Equilibrium; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 286, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
