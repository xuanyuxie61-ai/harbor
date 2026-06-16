# Reverse-Engineer Project 204: Sobol sensitivity analysis of a disk-shaped

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `204`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for Sobol sensitivity analysis of a disk-shaped is:

```text
synthesis-python-204 1.0
```

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in uncertainty quantification. The binary is meant for observation only; rebuild the behavior in your own files after probing it.

## Observable behavior

Reconstruct the command-line behavior for this uncertainty quantification target:

```text
Sobol sensitivity analysis of a disk-shaped
```

Start by matching these lines:

```text
PROJECT 204 : Sobol sensitivity analysis of a disk-shaped
geophysical reactor with chaotic mixing
and autocatalytic Lotka-Volterra kinetics.
Stage 0 / 7 : module smoke tests
[ok]   r8col_utils
[ok]   matrix_kernels
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 204 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
