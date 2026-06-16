# Computational Astrophysics

Project 251 asks for a faithful external clone of a synthesized computational plasma physics report generator.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. Keep runtime bounded and deterministic; expensive simulations can be replaced by small calibrated calculations.

## Workflow outline

- Unified entry point for the shearing-box MHD simulation.
- executes the full pipeline:
- 1. Load the default physical / numerical parameters.
- 2. Build a shearing-box grid (with bisection-selected refinement).

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

Version output:

```text
synthesis-python-251 1.0
```

## Stable landmarks

```text
PROJECT 251 - Computational Astrophysics
Accretion-disk MHD: high-order finite differences
and von Neumann stability analysis (small reproducible box)
Nx, Ny, Nz     = 32, 32, 8
[1/9] Loading default physical & numerical parameters ...
[2/9] Building shearing-box grid (bisection-selected refinement) ...
Lx, Ly, Lz (H) = 0.5, 0.5, 0.25
cells / MRI    = 1.87
aspect ratio   = 2.000
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Implement the flags explicitly instead of relying on argparse defaults that may format help differently.
