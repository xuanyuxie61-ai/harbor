# Reverse-Engineer Project 274: Electron-Phonon Coupling and Tc Prediction

## Build goal

You are in `/app/workspace` with a reference executable for project `274`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 274:

```text
synthesis-python-274 1.0
```

The benchmark centers on a scientific driver in lattice field theory, with behavior exposed through a local binary. A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

## What must match

Project 274 should be rebuilt around:

```text
Electron-Phonon Coupling and Tc Prediction
```

Useful observation anchors from the oracle:

```text
PROJECT 274 -- Electron-Phonon Coupling and Tc Prediction
High-order finite differences, stability analysis,
small-scale reproducible experiment.
Pipeline execution
[pipeline] Stage 1 : adaptive k-point mesh        ... OK  (0.198s)
[pipeline] Stage 2 : Brillouin zone               ... OK  (0.003s)
```

Finish by making a standalone workspace executable for Electron-Phonon Coupling and Tc Prediction; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 274, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
