# Stochastic Cahn-Hilliard ME-gPC CLI

This workspace contains a reference executable for a ProgramBench-style
reconstruction task. Rebuild a Python codebase that reproduces the executable's
observable command-line behavior.

## Reference binary

```bash
./executable
```

You may run this binary to inspect behavior. During verification, an unchanged
copy of this exact binary is removed before your build script is run.

## Required commands

```bash
./executable --version
./executable config --json
./executable basis --degree 3 --json
./executable quadrature --level 2 --json
./executable owen --h 1 --a 0.5 --json
./executable smoke --json
./executable run
```

`--version` prints `executable 201.1.0`.

`config --json` returns the default scientific model configuration. Important
fields include:

- `basis.max_degree = 4`
- `basis.truncation = "total_order"`
- `cahn_hilliard.nx = 32`
- `cahn_hilliard.ny = 32`
- `cahn_hilliard.dt = 0.0001`
- `cahn_hilliard.n_steps = 200`
- random variables: `kappa`, `gamma`, `mobility`
- `total_basis_dim = 35`

`basis --degree 3 --json` returns `n_dim = 3` and `n_basis = 20`.

`quadrature --level 2 --json` returns `n_points = 25` and a weight sum close to
`1.0`.

`owen --h 1 --a 0.5 --json` returns value
`0.043064691060608055`.

`smoke --json` runs a small deterministic setup check. It should report:

- `basis_dim = 10`
- `grid = [8, 8]`
- `sparse_points = 25`
- `steps = 2`
- `mass_error = 0.0`
- final energy less than initial energy

## Build contract

Create `compile.sh` or `build.sh` in `/app/workspace`. The script must generate
an executable file named `executable` in the same directory.

For a short, non-complete implementation guide, see `IMPLEMENTATION_NOTES.md`.
