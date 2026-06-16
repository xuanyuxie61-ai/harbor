# SOLAR FLARE MAGNETIC RECONNECTION SIMULATION

Project 262 asks for a faithful external clone of a synthesized computational plasma physics report generator.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. Keep runtime bounded and deterministic; expensive simulations can be replaced by small calibrated calculations.

## What to reproduce

- Unified entry point for the solar flare magnetic reconnection simulation.
- This module orchestrates the complete simulation pipeline:
- 1. Initialize plasma parameters and mesh
- 2. Set up Harris current sheet equilibrium

## Command surface

```bash
./executable --help
./executable --version
./executable
```

The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

Expected `--version` text:

```text
synthesis-python-262 1.0
```

## Output cues

```text
SOLAR FLARE MAGNETIC RECONNECTION SIMULATION
High-Order Finite Differences & Stability Analysis
Small-Scale Reproducible Experiment
SOLAR CORONA PLASMA PARAMETERS
[Phase 1] Plasma Parameters and Mesh Setup
B0 (upstream)         = 20.00 G
L_cs (half-thickness) = 5.00e+06 m
n0 (density)          = 1.00e+15 m^-3
T0 (temperature)      = 5.00e+06 K
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Implement the flags explicitly instead of relying on argparse defaults that may format help differently.
