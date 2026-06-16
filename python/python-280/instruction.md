# Reverse-Engineer Project 280: Multi-Scale Material Damage Evolution

## Build goal

You are in `/app/workspace` with a reference executable for project `280`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 280:

```text
synthesis-python-280 1.0
```

The program under observation is a deterministic numerical-methods benchmark demonstrator packaged as a single executable. Capture enough of the ordinary run to reproduce both the scientific storyline and the small numeric landmarks.

## What must match

Project 280 should be rebuilt around:

```text
Multi-Scale Material Damage Evolution
```

Useful observation anchors from the oracle:

```text
Project 280: Multi-Scale Material Damage Evolution
High-Order Finite Differences & Stability Analysis
Computational Materials Science — PhD-Level Simulation
Phase 1: Grid Generation with Crack-Tip Refinement
Domain: [0.0, 0.3] x [0.0, 0.3] m
Grid: 60 x 60 = 3600 nodes
```

Finish by making a standalone workspace executable for Multi-Scale Material Damage Evolution; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 280, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
