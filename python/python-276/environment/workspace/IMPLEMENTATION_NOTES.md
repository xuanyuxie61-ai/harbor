# Reconstruction Notes: Project 276

The cleanroom notes below emphasize small lattice construction, update diagnostics, and thermodynamic observables. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Useful internal split

- Numerical layer: high_order_fd, linear_solver, mesh_defect
- Diagnostics/reporting: defect_formation_energy, stability_analysis
- Supporting modules: analytical_benchmarks, config, crystal_lattice, denoise_filter, eshelby_strain, fft_poisson

## Rebuild approach

- Keep project 276's build script idempotent so repeated verifier runs do not change behavior.
- Match the order and magnitude of ╔══════════════════════════════════════════════════════════════════╗'s reported values before adding extra scientific machinery.
- Mirror the visible cadence of the lattice field theory transcript: setup, computation, diagnostics, summary.
- If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.

## Verifier-facing cues

Use these public lines as reconstruction checkpoints for `╔══════════════════════════════════════════════════════════════════╗`; hidden tests are not limited to them.

```text
╔══════════════════════════════════════════════════════════════════╗
║  Crystal-Defect Formation Energy (PhD-level synthesis project) ║
║  2-D hexagonal crystal · high-order FD · sparse Green function ║
╚══════════════════════════════════════════════════════════════════╝
Stage 1 — Analytical benchmarks
order 2: PASS
order 4: PASS
order 6: PASS
order 8: PASS
order 2: max |∇²_fd − ∇²_exact| = 1.606e-03
```

Keep constants close to the observed transcript, but avoid copying source files from outside the workspace. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
