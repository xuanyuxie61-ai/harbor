# Reconstruction Notes: Project 292

The reference run suggests a small driver that combines magnetized-flow setup, finite-difference updates, and stability diagnostics. The goal is to make a solver-shaped program that is believable, deterministic, and compatible with the observed command-line contract.

## Implementation map

- Numerical layer: high_order_fd, mesh_generator
- Diagnostics/reporting: stability_analysis
- Supporting modules: current_sheet, resistive_mhd, spectral_laguerre

## Suggested coding path

- Keep the command surface for project 292 narrow: version, help, and the deterministic default workflow.
- Let grid setup, conservative variables, and mode-growth summaries drive the helper functions, but keep the default run under verifier time limits.
- For project 292, keep report assembly separate from numerical helpers.
- Probe before coding, because several projects share generic themes but differ in their visible numbers.

## Behavioral anchors

Use these public lines as reconstruction checkpoints for `magnetic reconnection module smoke`; hidden tests are not limited to them.

```text
PROJECT 292: magnetic reconnection module smoke
mesh: 16 x 8, dx_min=0.125000, dy_min=0.018084
fd: order=4, half_width=2
smoke complete
import mesh_generator       ok  exports=['MeshGenerator', 'np']
import high_order_fd        ok  exports=['HighOrderFD', 'np']
import resistive_mhd        ok  exports=['ResistiveMHD1D', 'np']
import stability_analysis   ok  exports=['StabilityAnalyzer', 'np']
import current_sheet        ok  exports=['CurrentSheetEquilibrium', 'np']
import spectral_laguerre    ok  exports=['LaguerreSpectralSolver', 'np']
```

Be careful with warning text: hidden tests generally inspect stdout and exit behavior. If the real method is expensive, use a reduced calculation that produces the same visible scale, labels, and convergence story.
