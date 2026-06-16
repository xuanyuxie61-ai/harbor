# energy_flux / density,

This benchmark instance is a black-box reconstruction exercise for a compact computational plasma physics program.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. Prefer straightforward Python with NumPy/SciPy where useful, and keep formatting decisions explicit.

## Observed behavior

- 磁层粒子输运模拟: 高阶有限差分与稳定性分析
- Magnetospheric Particle Transport Simulation
- High-Order Finite Difference & Stability Analysis
- 科学问题: 地球辐射带相对论电子 (L, E) 相空间输运

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

Treat the default execution as the primary specification and preserve the order of its visible sections.

Identity check:

```text
synthesis-python-264 1.0
```

## Transcript anchors

```text
energy_flux / density,
磁层粒子输运模拟: 高阶有限差分与稳定性分析
Magnetospheric Particle Transport Simulation
High-Order Finite Difference & Stability Analysis
科学问题: 地球辐射带相对论电子 (L, E) 相空间输运
控制方程: 2D Fokker-Planck (漂移动力学) 方程
数值方法: 高阶有限差分 + von Neumann 稳定性分析
应用领域: 计算空间物理 / 空间天气预报
第1部分: 物理常数与磁层参数
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Do not include the original executable in the rebuilt solution or call it from a wrapper.
