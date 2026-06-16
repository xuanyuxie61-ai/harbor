# Reverse-Engineer Project 208: MULTI-FIDELITY UNCERTAINTY QUANTIFICATION PIPELINE

## Build goal

You are in `/app/workspace` with a reference executable for project `208`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 208:

```text
synthesis-python-208 1.0
```

The benchmark centers on a scientific driver in uncertainty quantification, with behavior exposed through a local binary. A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

## What must match

Project 208 should be rebuilt around:

```text
MULTI-FIDELITY UNCERTAINTY QUANTIFICATION PIPELINE
```

Useful observation anchors from the oracle:

```text
MULTI-FIDELITY UNCERTAINTY QUANTIFICATION PIPELINE
Protoplanetary-disk chemistry with autoregressive GP fusion
Python 3.11.11
1. Physical-system initialization
Reference disk state assembled at t = 0.5 Myr
QoI (column CO abundance) = 9.543e+27
```

Finish by making a standalone workspace executable for MULTI-FIDELITY UNCERTAINTY QUANTIFICATION PIPELINE; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 208, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
