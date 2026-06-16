# Reverse-Engineer Project 205: ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING

## Workspace objective

You are in `/app/workspace` with a reference executable for project `205`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

`./executable --version` identifies project 205 as:

```text
synthesis-python-205 1.0
```

The reference executable is the oracle for a reduced uncertainty quantification experiment with stable printed diagnostics. The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

## Public interface

The default run presents the following workflow title:

```text
ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING
```

Initial public cues for ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING:

```text
ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING
for Uncertainty Quantification in
Coupled Laser-Reaction-Diffusion Systems
STEP 1: Parameter Definition and Distributions
Dimension d = 4
Parameters: ['D_u', 'D_v', 'alpha', 'beta']
```

Create source files for ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING and provide `compile.sh` or `build.sh`; the script must leave `/app/workspace/executable` executable.

The final answer for ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING must be original source plus a build script, not a wrapper around the oracle. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
