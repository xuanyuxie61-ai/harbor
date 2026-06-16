# Tokamak Grad-Shafranov Equilibrium

This benchmark instance is a black-box reconstruction exercise for a compact computational plasma physics program.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. Prefer straightforward Python with NumPy/SciPy where useful, and keep formatting decisions explicit.

## What to reproduce

- PROJECT_286 — main.py
- Tokamak Grad-Shafranov equilibrium with high-order finite differences,
- MHD stability analysis, and multi-physics diagnostics.
- This is the unified entry point; no command-line arguments are required.

## Command surface

```bash
./executable --help
./executable --version
./executable
```

Treat the default execution as the primary specification and preserve the order of its visible sections.

Expected `--version` text:

```text
synthesis-python-286 1.0
```

## Output cues

```text
PROJECT_286: Tokamak Grad-Shafranov Equilibrium
High-order finite differences + MHD stability analysis
(small-scale reproducible experiment)
1. Geometry and plasma profiles
κ = 1.7,  δ = 0.33
grid: Nr=41, Nz=41,  dR=0.0275, dZ=0.0300
pressure amp  c_1 = 40000.0 Pa
FF'  amp      d_1 = 1.5 T²
special-function check:
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Do not include the original executable in the rebuilt solution or call it from a wrapper.
