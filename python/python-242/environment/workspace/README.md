# Nuclear shell-model energy levels and transition

This benchmark instance is a black-box reconstruction exercise for a compact fusion and radiation transport program.

The program reads like a research demonstration: it sets up mesh data, cross sections, and solver diagnostics, then prints a staged report with deterministic diagnostics. Prefer straightforward Python with NumPy/SciPy where useful, and keep formatting decisions explicit.

## Scientific role

- probabilities with high-order finite differences and stability
- analysis (small-scale, reproducible experiment)
- Step 0: Problem specification
- Nucleus         : 18O (Z=8, N=10)

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

Treat the default execution as the primary specification and preserve the order of its visible sections.

Stable version text:

```text
synthesis-python-242 1.0
```

## Report signatures

```text
PROJECT 242: Nuclear shell-model energy levels and transition
probabilities with high-order finite differences and stability
analysis (small-scale, reproducible experiment)
Step 0: Problem specification
Nucleus         : 18O (Z=8, N=10)
Core            : 16O (Z=8, N=8)
Valence neutrons: 2
Model space     : sd-shell
Orbitals        : 1d_{5/2}, 2s_{1/2}, 1d_{3/2}
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Do not include the original executable in the rebuilt solution or call it from a wrapper.
