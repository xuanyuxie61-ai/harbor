# Reverse-Engineer Project 271: 1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis

## Workspace objective

You are in `/app/workspace` with a reference executable for project `271`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

`./executable --version` identifies project 271 as:

```text
synthesis-python-271 1.0
```

The reference executable is the oracle for a reduced lattice field theory experiment with stable printed diagnostics. The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

## Public interface

The default run presents the following workflow title:

```text
1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis
```

Initial public cues for 1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis:

```text
(small-scale reproducible experiment, Python)
Stage 1 : NAS-style benchmark kernels
btrix                  residual = 1.619e-16
cfft2d                 residual = 1.361e-15
PROJECT 271 : 1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis
cholsky                residual = 1.349e-16
```

Create source files for 1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis and provide `compile.sh` or `build.sh`; the script must leave `/app/workspace/executable` executable.

The final answer for 1D TFIM quantum phase transition, high-order FD, finite-size scaling, stability analysis must be original source plus a build script, not a wrapper around the oracle. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
