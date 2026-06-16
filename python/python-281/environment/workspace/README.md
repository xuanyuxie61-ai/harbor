# 电池电极材料离子扩散模拟

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in lattice field theory.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. A compact implementation is acceptable when it keeps the same CLI, exit status, section order, and recognizable diagnostics.

## Workflow outline

- 统一入口: 电池电极材料离子扩散模拟
- 高阶有限差分与稳定性分析 (小规模可复现实验)
- PROJECT_281 合成项目
- 融合种子项目:

## Reference runs

```bash
./executable --help
./executable --version
./executable
```

The binary is meant for observation only; rebuild the behavior in your own files after probing it.

Version output:

```text
synthesis-python-281 1.0
```

## Stable landmarks

```text
PROJECT 281: 电池电极材料离子扩散模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
博士级计算材料科学合成项目
阶段 1: 物理常数与 Arrhenius 扩散系数
T [K]        D [m²/s]    V_T [mV]
────────  ──────────────  ──────────
273.15      8.2721e-15     23.5382
298.15      3.9000e-14     25.6926
323.15      1.4465e-13     27.8469
```

## Build output

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Be careful with warning text: hidden tests generally inspect stdout and exit behavior.
