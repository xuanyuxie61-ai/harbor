# DUSTY PLASMA CRYSTAL SIMULATION

This benchmark instance is a black-box reconstruction exercise for a compact lattice field theory program.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. Prefer straightforward Python with NumPy/SciPy where useful, and keep formatting decisions explicit.

## Visible purpose

- Unified entry point for the dusty plasma crystal simulation.
- Scientific problem:
- Simulation of 2D dusty plasma crystal structure using high-order
- finite difference methods and eigenvalue-based stability analysis.

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

Treat the default execution as the primary specification and preserve the order of its visible sections.

Reference version:

```text
synthesis-python-297 1.0
```

## Reference cues

```text
DUSTY PLASMA CRYSTAL SIMULATION
High-Order Finite Difference & Stability Analysis
Plasma regime: VALID
Crystal regime            = INTERMEDIATE
[Step 1] Initializing dusty plasma regime...
n_e [m^-3]                = 1.000e+15
T_e [eV]                  = 2.500
T_i [eV]                  = 0.0300
lambda_D [m]              = 4.048e-05
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Do not include the original executable in the rebuilt solution or call it from a wrapper.
