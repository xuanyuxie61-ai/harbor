# CMB POWER SPECTRUM ESTIMATION PIPELINE

This case exposes the observable surface of a research-style astrophysics simulation driver rather than its source.

The program reads like a research demonstration: it sets up model parameters, evolution loops, and summary tables, then prints a staged report with deterministic diagnostics. Avoid delegating to the supplied executable; tests remove unchanged reference binaries before building.

## Workflow outline

- Unified entry point for the CMB power-spectrum estimation pipeline.
- Scientific goal:
- Estimate the angular power spectrum C_l of the CMB temperature field
- using a discretised Laplace-Beltrami operator on a cubed-sphere mesh,

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

Version output:

```text
synthesis-python-257 1.0
```

## Stable landmarks

```text
#                                                                      #
#   CMB POWER SPECTRUM ESTIMATION PIPELINE                            #
#   High-Order Finite Differences & Stability Analysis                #
#   (Small-Scale Reproducible Experiment)                             #
CMB POWER SPECTRUM ESTIMATION PIPELINE
High-Order Finite Differences & Stability Analysis
(Small-Scale Reproducible Experiment)
Preliminary:  Planck 2018 derived parameters
theta_*    = 3.683033e-23
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.
