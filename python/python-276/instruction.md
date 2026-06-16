# Reverse-Engineer Project 276: ╔══════════════════════════════════════════════════════════════════╗

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `276`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for ╔══════════════════════════════════════════════════════════════════╗ is:

```text
synthesis-python-276 1.0
```

The visible binary represents a synthesized lattice field theory workflow with a deterministic command-line transcript. Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

## Observable behavior

Reconstruct the command-line behavior for this lattice field theory target:

```text
╔══════════════════════════════════════════════════════════════════╗
```

Start by matching these lines:

```text
╔══════════════════════════════════════════════════════════════════╗
║  Crystal-Defect Formation Energy (PhD-level synthesis project) ║
║  2-D hexagonal crystal · high-order FD · sparse Green function ║
╚══════════════════════════════════════════════════════════════════╝
Stage 1 — Analytical benchmarks
order 2: PASS
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 276 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
