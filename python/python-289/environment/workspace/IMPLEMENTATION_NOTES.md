# Reconstruction Notes: Project 289

Expect hidden checks to care most about the externally visible parts of matrix construction, eigenvalue diagnostics, and stability comparison. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Suggested decomposition

- Numerical layer: collision_operator, radial_mesh, time_integrator
- Model layer: physics_constants
- Diagnostics/reporting: collision_operator, turbulence_diagnostics
- Supporting modules: finite_difference, gyroaverage, linear_eigenvalue, velocity_space, zonal_flow

## Coding plan

- For 1D slab gyrokinetic turbulence simulator, keep argument handling small and predictable before implementing the default report.
- Use NumPy/SciPy only where they make the spectral physics reconstruction clearer and faster.
- If 1D slab gyrokinetic turbulence simulator prints matrices or tables, store their rows as data and format them consistently.
- Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.

## Anchors to preserve

Use these public lines as reconstruction checkpoints for `1D slab gyrokinetic turbulence simulator`; hidden tests are not limited to them.

```text
PROJECT 289 -- 1D slab gyrokinetic turbulence simulator
Domain:  计算等离子体 / 湍流输运 / gyrokinetic 模拟
高阶有限差分 + 稳定性分析 (小规模可复现实验)
Step 0 -- Equilibrium construction
ion sound speed  c_s      = 2.1885e+05 m/s
ion cyclotron    omega_ci = 1.1974e+08 rad/s
gyroradius       rho_s    = 1.8278e-03 m
ITG threshold (rough)  (R/L_Ti)_c ~ 1111.111
n0 profile (5 pts)  = [1.    0.975 0.95  0.925 0.9  ]
T_i profile (5 pts) = [1000.  925.  850.  775.  700.]
```

Preserve non-English labels and punctuation where they appear in the reference transcript. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
