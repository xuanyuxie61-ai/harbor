# Reverse-Engineer Project 255: 系外行星大气光谱反演

## Black-box target

You are in `/app/workspace` with a reference executable for project `255`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Use this version text to confirm you are probing the right oracle:

```text
synthesis-python-255 1.0
```

Project 255 is framed as a cleanroom reproduction task around spectral physics. The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

## Reference behavior

The binary's scientific topic is:

```text
系外行星大气光谱反演
```

The transcript begins to define the target through:

```text
PROJECT 255 : 系外行星大气光谱反演
Exoplanet Atmospheric Spectral Retrieval
High-Order Finite Differences & Stability Analysis
Step 1: 大气结构建模 (WASP-39b 型热木星)
行星质量        : 5.314e+26 kg
行星半径        : 9.079e+07 m
```

Your rebuilt spectral physics program must include a build step, either `compile.sh` or `build.sh`, that creates `/app/workspace/executable`.

Do not use internet access or outside source recovery for 系外行星大气光谱反演; infer behavior from the local artifacts. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
