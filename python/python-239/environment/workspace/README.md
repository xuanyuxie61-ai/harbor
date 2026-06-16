# QGP hydrodynamics module smoke

This is a ProgramBench-style task: infer and reproduce a computational plasma physics CLI from documentation plus black-box runs.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. The verifier rewards the public interface: executable creation, flags, deterministic output, and selected anchors.

## Workflow outline

- main.py  —  Unified Entry Point for the QGP Hydrodynamics Project
- Drives the complete computational pipeline for a small-scale reproducible
- experiment on heavy-ion collision quark-gluon plasma hydrodynamics:
- 1. Initialise the QGP fireball (Glauber + hot-spot model).

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

Start with the version and help flags, then run the binary without arguments to collect the full staged report.

Version output:

```text
synthesis-python-239 1.0
```

## Stable landmarks

```text
PROJECT 239: QGP hydrodynamics module smoke
import qgp_config                   ok
import qgp_eos                      ok
import qgp_grid                     ok
import qgp_initial_conditions       ok
import qgp_operators                ok
import qgp_time_integration         ok
import qgp_stability_analysis       ok
import qgp_flow_harmonics           ok
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep the generated executable at the workspace root and make it executable.
