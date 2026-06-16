# Reverse-Engineer Project 283: Perovskite Solar-Cell Defect-State Calculator

## Workspace objective

You are in `/app/workspace` with a reference executable for project `283`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

`./executable --version` identifies project 283 as:

```text
synthesis-python-283 1.0
```

This is a ProgramBench-style task: infer and reproduce a numerical-methods benchmark CLI from documentation plus black-box runs. Start with the version and help flags, then run the binary without arguments to collect the full staged report.

## Public interface

The default run presents the following workflow title:

```text
Perovskite Solar-Cell Defect-State Calculator
```

Initial public cues for Perovskite Solar-Cell Defect-State Calculator:

```text
PROJECT 283: Perovskite Solar-Cell Defect-State Calculator
High-Order Finite Differences + Stability Analysis
Device length: 500.0 nm
Built-in voltage: 1.00 V
FD order (2p): 12
Defect density (default): 1.00e+21 m^-3
```

Create source files for Perovskite Solar-Cell Defect-State Calculator and provide `compile.sh` or `build.sh`; the script must leave `/app/workspace/executable` executable.

The final answer for Perovskite Solar-Cell Defect-State Calculator must be original source plus a build script, not a wrapper around the oracle. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
