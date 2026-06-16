# Reconstruction Notes: Project 275

The binary behaves like a scripted experiment focused on matrix construction, eigenvalue diagnostics, and stability comparison. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Component hints

- Numerical layer: adaptive_ep_mesh, brillouin_quadrature, high_order_fd, spectral_solver
- Diagnostics/reporting: exceptional_point_locator, parameter_continuation, spectral_statistics, stability_analysis
- Supporting modules: nonhermitian_hamiltonian, phase_classifier

## Implementation strategy

- Use a tiny dispatch layer for Non-Hermitian Spectral Structure & Exceptional; hidden tests should not depend on accidental framework formatting.
- If a real spectral physics solver would be expensive, replace it with a reduced calculation that tells the same story.
- Do not add banners or debug text around Non-Hermitian Spectral Structure & Exceptional; hidden checks may parse stdout.
- Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.

## Public report cues

Use these public lines as reconstruction checkpoints for `Non-Hermitian Spectral Structure & Exceptional`; hidden tests are not limited to them.

```text
PROJECT 275: Non-Hermitian Spectral Structure & Exceptional
Points - High-Order Finite Difference & Stability Analysis
1. Non-Hermitian SSH Hamiltonian Construction
SSH parameters: t1=1.0, t2=0.6, gamma=0.35, k=1.0472
H_SSH(k) =
[[0. +0.35j       1.3-0.51961524j]
[1.3+0.51961524j 0. -0.35j      ]]
Discriminant Delta(k) = 7.350000 + 0.000000i
|Delta| = 7.350000e+00
Hatano-Nelson: N=12, g=0.4
```

Do not include the original executable in the rebuilt solution or call it from a wrapper. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
