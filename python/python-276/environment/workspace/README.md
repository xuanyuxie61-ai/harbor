# ╔══════════════════════════════════════════════════════════════════╗

The visible binary represents a synthesized lattice field theory workflow with a deterministic command-line transcript.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. The hidden tests may look beyond the examples, so preserve the general report shape rather than only one line.

## Observed behavior

- main.py — Unified entry point for the crystal-defect formation energy project
- Run this file with no arguments to perform the full computation:
- The pipeline proceeds in 9 stages:
- Stage 1. Validate the numerical kernels against analytical benchmarks

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

Identity check:

```text
synthesis-python-276 1.0
```

## Transcript anchors

```text
╔══════════════════════════════════════════════════════════════════╗
║  Crystal-Defect Formation Energy (PhD-level synthesis project) ║
║  2-D hexagonal crystal · high-order FD · sparse Green function ║
╚══════════════════════════════════════════════════════════════════╝
Stage 1 — Analytical benchmarks
order 2: PASS
order 4: PASS
order 6: PASS
order 8: PASS
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.
