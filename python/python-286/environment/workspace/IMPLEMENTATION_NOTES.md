# Reconstruction Notes: Project 286

The binary behaves like a scripted experiment focused on magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: nonlinear_solver, sparse_operators
- Model layer: field_line_tracer, transport_pdf
- Diagnostics/reporting: boundary_reconstruction, diagnostics, pic_reconstruction, stability_analysis
- Supporting modules: flux_surface_integrals, grad_shafranov, plasma_profiles, quasi_mc, tokamak_geometry

## Suggested coding path

- Use a tiny dispatch layer for Tokamak Grad-Shafranov Equilibrium; hidden tests should not depend on accidental framework formatting.
- If a real computational plasma physics solver would be expensive, replace it with a reduced calculation that tells the same story.
- Do not add banners or debug text around Tokamak Grad-Shafranov Equilibrium; hidden checks may parse stdout.
- Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `Tokamak Grad-Shafranov Equilibrium`; hidden tests are not limited to them.

```text
PROJECT_286: Tokamak Grad-Shafranov Equilibrium
High-order finite differences + MHD stability analysis
(small-scale reproducible experiment)
1. Geometry and plasma profiles
κ = 1.7,  δ = 0.33
grid: Nr=41, Nz=41,  dR=0.0275, dZ=0.0300
pressure amp  c_1 = 40000.0 Pa
FF'  amp      d_1 = 1.5 T²
special-function check:
Ci(1.0) = 0.337404   (ref ≈ 0.337404)
```

Do not include the original executable in the rebuilt solution or call it from a wrapper. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
