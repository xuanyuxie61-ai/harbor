# MULTI-FIDELITY UNCERTAINTY QUANTIFICATION PIPELINE

The benchmark centers on a scientific driver in uncertainty quantification, with behavior exposed through a local binary.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. The target is not source recovery. It is an original implementation that reproduces the black-box contract.

## What to reproduce

- Unified entry point for the multi-fidelity uncertainty-quantification
- project.  Zero-argument invocation:
- $ python main.py
- Performs the full pipeline:

## Command surface

```bash
./executable --help
./executable --version
./executable
```

A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

Expected `--version` text:

```text
synthesis-python-208 1.0
```

## Output cues

```text
MULTI-FIDELITY UNCERTAINTY QUANTIFICATION PIPELINE
Protoplanetary-disk chemistry with autoregressive GP fusion
Python 3.11.11
1. Physical-system initialization
Reference disk state assembled at t = 0.5 Myr
QoI (column CO abundance) = 9.543e+27
2. Fidelity hierarchy initialization
Level 0: Analytic, tags=['power-law', 'steady-state', 'no-planet', 'isothermal']
Level 1: Surrogate, tags=['lagrange', 'sparse-grid', 'no-planet', 'thermal']
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make the no-argument path finish quickly under verifier time limits.
