# Perovskite Solar-Cell Defect-State Calculator

This is a ProgramBench-style task: infer and reproduce a numerical-methods benchmark CLI from documentation plus black-box runs.

The program reads like a research demonstration: it sets up stencil coefficients, amplification factors, and convergence indicators, then prints a staged report with deterministic diagnostics. The verifier rewards the public interface: executable creation, flags, deterministic output, and selected anchors.

## Program sketch

- Unified entry point for the PROJECT-283 perovskite solar-cell defect-state
- calculation code. Runs the complete pipeline:
- [1] Sanity checks on physical constants and high-order FD stencils
- [2] Mesh generation, quality check, and CVT optimization

## Observation commands

```bash
./executable --help
./executable --version
./executable
```

Start with the version and help flags, then run the binary without arguments to collect the full staged report.

Version probe:

```text
synthesis-python-283 1.0
```

## Observable anchors

```text
PROJECT 283: Perovskite Solar-Cell Defect-State Calculator
High-Order Finite Differences + Stability Analysis
Device length: 500.0 nm
Built-in voltage: 1.00 V
FD order (2p): 12
Defect density (default): 1.00e+21 m^-3
1. Physical constants sanity check
V_t [mV]               = 25.852
N_c [1/m^3]            = 1.043e+24
```

## Build expectation

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep the generated executable at the workspace root and make it executable.
