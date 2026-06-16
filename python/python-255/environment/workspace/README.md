# 系外行星大气光谱反演

Project 255 is framed as a cleanroom reproduction task around spectral physics.

The program reads like a research demonstration: it sets up operators, spectra, and formatted numerical landmarks, then prints a staged report with deterministic diagnostics. Use fixed seeds and stable constants where the reference advertises reproducibility.

## Visible purpose

- PROJECT 255 : 系外行星大气光谱反演 —— 高阶有限差分与稳定性分析
- 一、项目定位
- 本项目围绕 **计算天体物理** 领域的核心前沿问题: **系外行星大气光谱反演**。
- 针对热木星 (Hot Jupiter, 以 WASP-39b 为典型代表) 类同步自转系外行星,

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

The executable is deterministic, so repeated probes should agree apart from environment-specific warning streams.

Reference version:

```text
synthesis-python-255 1.0
```

## Reference cues

```text
PROJECT 255 : 系外行星大气光谱反演
Exoplanet Atmospheric Spectral Retrieval
High-Order Finite Differences & Stability Analysis
Step 1: 大气结构建模 (WASP-39b 型热木星)
行星质量        : 5.314e+26 kg
行星半径        : 9.079e+07 m
表面重力        : 4.303 m/s^2
平衡温度        : 1116.0 K
标高 (T_eq)     : 9.303e+11 m
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. If a full solver would be slow, use reduced arrays or closed-form summaries calibrated to the observed output.
