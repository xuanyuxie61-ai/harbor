# Reverse-Engineer Project 249: Stellar evolution with coupled nuclear networks

## Black-box target

You are in `/app/workspace` with a reference executable for project `249`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Use this version text to confirm you are probing the right oracle:

```text
synthesis-python-249 1.0
```

The reference executable is the oracle for a reduced astrophysics simulation experiment with stable printed diagnostics. The simplest path is to implement the reporting flow first, then add only the numerical detail needed to support it.

## Reference behavior

The binary's scientific topic is:

```text
Stellar evolution with coupled nuclear networks
```

The transcript begins to define the target through:

```text
PROJECT_249 - Stellar evolution with coupled nuclear networks
High-order finite difference + adaptive mesh + stiff integrators
Stage 1 : Adaptive mass grid construction (1D CVT)
max/min spacing ratio = 2.650
smoothness measure    = 0.635
first 6 mass coords   = ['0.000e+00', '4.208e+32', '1.143e+33', '1.887e+33', '2.660e+33', '3.530e+33']
```

Your rebuilt astrophysics simulation program must include a build step, either `compile.sh` or `build.sh`, that creates `/app/workspace/executable`.

Do not use internet access or outside source recovery for Stellar evolution with coupled nuclear networks; infer behavior from the local artifacts. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
