# Reverse-Engineer Project 292: magnetic reconnection module smoke

## Build goal

You are in `/app/workspace` with a reference executable for project `292`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 292:

```text
synthesis-python-292 1.0
```

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in computational plasma physics. The binary is meant for observation only; rebuild the behavior in your own files after probing it.

## What must match

Project 292 should be rebuilt around:

```text
magnetic reconnection module smoke
```

Useful observation anchors from the oracle:

```text
PROJECT 292: magnetic reconnection module smoke
mesh: 16 x 8, dx_min=0.125000, dy_min=0.018084
fd: order=4, half_width=2
smoke complete
import mesh_generator       ok  exports=['MeshGenerator', 'np']
import high_order_fd        ok  exports=['HighOrderFD', 'np']
```

Finish by making a standalone workspace executable for magnetic reconnection module smoke; hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 292, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
