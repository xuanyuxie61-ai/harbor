# Task: Rebuild a Program From Its Binary and Documentation

## Task Context

You are given a cleanroom workspace for a ProgramBench-style reverse-engineering task.

The workspace is:

```text
/app/workspace
```

At the start, it contains:

```text
/app/workspace/executable
/app/workspace/README.md
/app/workspace/COPYING
```

`./executable` is the original reference binary. `README.md` is the bundled user-facing program documentation. You must use these materials to infer the program's externally observable behavior.

## Your Task

Implement an original Python codebase from scratch that reproduces the behavior of the reference executable.

Your final workspace must contain a build script:

```text
compile.sh
```

or:

```text
build.sh
```

The build script must create an executable file at:

```text
/app/workspace/executable
```

The generated `executable` must be runnable directly from the workspace root.

For a Python implementation, a valid `compile.sh` can create a shell wrapper around your `main.py`, for example:

```bash
#!/usr/bin/env bash
set -euo pipefail
cat > executable <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/main.py" "$@"
EOF
chmod +x executable
```

## Important Reverse-Engineering Rules

This is a reverse-engineering benchmark.

You must write original source code that reproduces the reference executable's behavior. The expected workflow is:

1. Read `README.md`.
2. Run `./executable` with many commands and inputs.
3. Observe stdout, stderr, exit codes, JSON fields, numeric values, and error behavior.
4. Design and implement a compatible Python program.
5. Write `compile.sh` or `build.sh`.
6. Verify that your generated `./executable` behaves like the reference binary.

Do not simply wrap or delegate to the provided reference `./executable`.

Do not copy the original reference binary into your final solution.

Do not rely on internet access, external repositories, package registries, or downloading the original project.

Do not assume hidden tests only cover the examples below. Hidden tests may probe variations of the documented CLI behavior.

You may use Python standard library modules and the scientific packages available in the environment, especially `numpy` and `scipy`.

## Program Overview

The reference program is a command-line scientific-computing tool for uncertainty quantification of a stochastic Cahn-Hilliard phase-field model using multi-element generalized polynomial chaos.

It exposes both small deterministic CLI probes and a longer full demonstration workflow.

Required command-line interface:

```bash
./executable --version
./executable config --json
./executable basis --degree 3 --json
./executable quadrature --level 2 --json
./executable owen --h 1 --a 0.5 --json
./executable smoke --json
./executable run
```

With no subcommand, the executable should run the full demonstration workflow, equivalent to `run`.

## Behavioral Targets

Your implementation should reproduce the observable behavior of the reference binary, including:

- command names and options
- JSON output structure
- important numeric values
- deterministic smoke-check behavior
- version string
- exit status for successful commands

Known reference points:

```bash
./executable --version
```

prints:

```text
executable 201.1.0
```

```bash
./executable basis --degree 3 --json
```

returns JSON with:

```text
n_dim = 3
n_basis = 20
truncation = "total_order"
```

```bash
./executable quadrature --level 2 --json
```

returns JSON with:

```text
n_points = 25
weight_sum ~= 1.0
```

```bash
./executable owen --h 1 --a 0.5 --json
```

returns JSON with:

```text
value ~= 0.043064691060608055
```

```bash
./executable smoke --json
```

returns JSON with:

```text
basis_dim = 10
grid = [8, 8]
sparse_points = 25
steps = 2
mass_error = 0.0
energy_final < energy_initial
```

Read `/app/workspace/README.md` for the program-level documentation, then use the reference binary to discover any additional details needed for compatibility.

