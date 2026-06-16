# Implementation Notes

This program can be rebuilt as a small multi-file Python CLI. A faithful
solution does not need a large framework, but it should separate numerical
components instead of putting every calculation in one function.

## Suggested Module Layout

The original implementation is organized around these responsibilities:

- `main.py`: command-line parser, JSON/text output, full workflow driver.
- `config.py`: dataclass-style default parameters for basis, random measures,
  sparse grid, multi-element settings, and the Cahn-Hilliard solver.
- `measure.py`: probability measures, Gaussian quadrature, CDF/PDF helpers,
  and Owen's T function.
- `utils.py`: multi-index enumeration, simple sparse-matrix helpers, line
  grids, and spectral derivative utilities.
- `grid.py`: physical mesh generation, initial condition generation, and
  sparse-grid node assembly.
- `polynomial_basis.py`: total-order polynomial chaos basis over three random
  variables.
- `sparse_grid.py`: adaptive/Smolyak-style sparse quadrature summaries.
- `cahn_hilliard.py`: deterministic Cahn-Hilliard solver used by `smoke`.

The longer `run` command may also include lightweight placeholders or compact
implementations for Galerkin projection, multi-element refinement, statistics,
multi-fidelity analysis, neural closure, and MCMC. The hidden tests focus on
observable CLI behavior, so prioritize deterministic command outputs first.

## Model Defaults

The stochastic inputs are three-dimensional:

1. `kappa`: Gaussian-like, mean `0.01`, standard deviation `0.002`
2. `gamma`: uniform on approximately `[0.8, 1.2]`
3. `mobility`: Gaussian-like, mean `1.0`, standard deviation `0.15`

The default polynomial basis uses total-order truncation with max degree `4`.
The default Cahn-Hilliard grid is `32 x 32`, with `dt = 1e-4` and `200` nominal
steps.

## CLI Behavior Hints

Implement both JSON and text modes where documented. JSON output should be
stable and easy to parse. NumPy scalars and arrays should be converted to plain
Python numbers/lists before serialization.

Useful combinatorial anchors:

- degree `0` in three dimensions has `1` total-order basis term
- degree `1` has `4`
- degree `2` has `10`
- degree `3` has `20`
- degree `4` has `35`

The first total-order indices are ordered like:

```text
[0, 0, 0], [0, 0, 1], [0, 0, 2], ...
```

Sparse quadrature should produce the following point counts:

- level `1`: `7`
- level `2`: `25`
- level `3`: `69`

Owen's T should be numerically accurate for common probes, especially:

```text
T(1, 0.5) ~= 0.043064691060608055
```

## Smoke Command

`smoke` should be deterministic. It constructs a small physical mesh, generates
a seeded random initial concentration field, runs a short deterministic
Cahn-Hilliard update, and reports mass and energy summaries.

The default smoke command uses degree `2`, sparse-grid level `2`, an `8 x 8`
mesh, and `2` time steps. It should conserve mass to numerical precision and
slightly reduce free energy.

## Difficulty Calibration

This task is intended to be solvable by a strong model that actively probes the
binary, but not by simply memorizing a few examples. A good solution should
combine:

- CLI reverse engineering
- structured JSON reproduction
- basic polynomial-chaos combinatorics
- sparse quadrature approximations
- Owen T numerical integration
- a compact deterministic Cahn-Hilliard smoke solver

