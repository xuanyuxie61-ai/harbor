# Reverse-Engineer Project 239: QGP hydrodynamics module smoke

## Cleanroom objective

You are in `/app/workspace` with a reference executable for project `239`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Project 239's identity check should return:

```text
synthesis-python-239 1.0
```

This is a ProgramBench-style task: infer and reproduce a computational plasma physics CLI from documentation plus black-box runs. Start with the version and help flags, then run the binary without arguments to collect the full staged report.

## Behavioral target

The public-facing subject of project 239 is:

```text
QGP hydrodynamics module smoke
```

Visible cues for the computational plasma physics workflow:

```text
PROJECT 239: QGP hydrodynamics module smoke
import qgp_config                   ok
import qgp_eos                      ok
import qgp_grid                     ok
import qgp_initial_conditions       ok
import qgp_operators                ok
```

The final workspace for project 239 needs your implementation and a build script that regenerates `./executable`.

Do not copy, unpack, wrap, or call the supplied binary while solving QGP hydrodynamics module smoke; create a fresh implementation. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
