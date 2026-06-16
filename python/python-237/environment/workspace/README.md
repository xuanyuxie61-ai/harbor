# Lattice QCD: Quark Propagator & Gauge Field Sampling

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in lattice field theory.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. A compact implementation is acceptable when it keeps the same CLI, exit status, section order, and recognizable diagnostics.

## Visible purpose

- main.py - Unified entry point for the Lattice QCD project.
- Project: Lattice QCD - Quark Propagator and Gauge Field Sampling
- High-Order Finite Differences and Stability Analysis
- (Small-Scale Reproducible Experiment)

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

The binary is meant for observation only; rebuild the behavior in your own files after probing it.

Reference version:

```text
synthesis-python-237 1.0
```

## Reference cues

```text
Lattice QCD: Quark Propagator & Gauge Field Sampling
High-Order Finite Differences and Stability Analysis
(Small-Scale Reproducible Experiment)
Step 1: Lattice Initialization
Lattice: 4^3 x 4 = 256 sites
Lattice spacing: a = 0.1 fm
Gauge coupling: beta = 2.5, g^2 = 1.6000
Bare quark mass: m0 = 0.1
Hopping parameter: kappa = 0.121951
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Be careful with warning text: hidden tests generally inspect stdout and exit behavior.
