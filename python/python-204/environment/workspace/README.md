# Sobol sensitivity analysis of a disk-shaped

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in uncertainty quantification.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. A compact implementation is acceptable when it keeps the same CLI, exit status, section order, and recognizable diagnostics.

## Observed behavior

- Unified entry point for the Sobol sensitivity analysis of the
- disk-shaped geophysical reactor.  Zero arguments:
- runs the complete pipeline:
- 1. Smoke test (``robustness.run_smoke_tests``).

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

The binary is meant for observation only; rebuild the behavior in your own files after probing it.

Identity check:

```text
synthesis-python-204 1.0
```

## Transcript anchors

```text
PROJECT 204 : Sobol sensitivity analysis of a disk-shaped
geophysical reactor with chaotic mixing
and autocatalytic Lotka-Volterra kinetics.
Stage 0 / 7 : module smoke tests
[ok]   r8col_utils
[ok]   matrix_kernels
[ok]   elliptic_green
[ok]   chaotic_mixing
[ok]   disk_monomial_integral
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Be careful with warning text: hidden tests generally inspect stdout and exit behavior.
