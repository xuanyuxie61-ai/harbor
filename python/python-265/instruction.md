# Reverse-Engineer Project 265: Cosmic-Ray Transport in the Heliosphere

## Workspace objective

You are in `/app/workspace` with a reference executable for project `265`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

`./executable --version` identifies project 265 as:

```text
synthesis-python-265 1.0
```

The visible binary represents a synthesized numerical-methods benchmark workflow with a deterministic command-line transcript. Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

## Public interface

The default run presents the following workflow title:

```text
Cosmic-Ray Transport in the Heliosphere
```

Initial public cues for Cosmic-Ray Transport in the Heliosphere:

```text
PROJECT 265 -- Cosmic-Ray Transport in the Heliosphere
High-order finite differences & stability analysis
EXPERIMENT 1 -- heliospheric mesh & Parker IMF
radial grid : Nr = 48, r_min = 0.050 AU, r_max = 120.0 AU
mu grid     : Nmu = 16, min = -0.9950, max = +0.9950
EXPERIMENT 2 -- high-order finite-difference stencils
```

Create source files for Cosmic-Ray Transport in the Heliosphere and provide `compile.sh` or `build.sh`; the script must leave `/app/workspace/executable` executable.

The final answer for Cosmic-Ray Transport in the Heliosphere must be original source plus a build script, not a wrapper around the oracle. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
