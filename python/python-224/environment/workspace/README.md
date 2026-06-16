# Higgs Decay Signal Strength Fitting

This case exposes the observable surface of a research-style numerical-methods benchmark driver rather than its source.

The program reads like a research demonstration: it sets up stencil coefficients, amplification factors, and convergence indicators, then prints a staged report with deterministic diagnostics. Avoid delegating to the supplied executable; tests remove unchanged reference binaries before building.

## Scientific role

- PROJECT_224 :: Computational HEP - Higgs Decay Signal Strength Fitting
- High-Order Finite Differences & Stability Analysis (Small-scale reproducible experiment)
- This is the unified entry point. It runs the full analysis pipeline:
- 1. SM constants & Higgs effective potential V(phi)

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

Stable version text:

```text
synthesis-python-224 1.0
```

## Report signatures

```text
PROJECT_224 :: Higgs Decay Signal Strength Fitting
High-Order Finite Differences & Stability Analysis
Computational High-Energy Physics (small-scale reproducible experiment)
sin^2 theta_W = 0.222897
Recovered VEV          : v = 230.900734 GeV  (input 246.2196)
V''(v)                 : 16116.0408 GeV^2   (m_h^2 = 15647.51)
lambda_eff(v)          : -0.104611
EW phase-transition    : barrier height = 9.6119e+07 GeV^4
Channels fitted : 5
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.
