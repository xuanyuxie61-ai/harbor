# Reverse-Engineer Project 251: Computational Astrophysics

## Cleanroom objective

You are in `/app/workspace` with a reference executable for project `251`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Project 251's identity check should return:

```text
synthesis-python-251 1.0
```

Project 251 asks for a faithful external clone of a synthesized computational plasma physics report generator. The no-argument run is the useful probe; the flags mainly establish the wrapper contract and identity string.

## Behavioral target

The public-facing subject of project 251 is:

```text
Computational Astrophysics
```

Visible cues for the computational plasma physics workflow:

```text
PROJECT 251 - Computational Astrophysics
Accretion-disk MHD: high-order finite differences
and von Neumann stability analysis (small reproducible box)
Nx, Ny, Nz     = 32, 32, 8
[1/9] Loading default physical & numerical parameters ...
[2/9] Building shearing-box grid (bisection-selected refinement) ...
```

The final workspace for project 251 needs your implementation and a build script that regenerates `./executable`.

Do not copy, unpack, wrap, or call the supplied binary while solving Computational Astrophysics; create a fresh implementation. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
