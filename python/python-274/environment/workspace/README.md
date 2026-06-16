# Electron-Phonon Coupling and Tc Prediction

The benchmark centers on a scientific driver in lattice field theory, with behavior exposed through a local binary.

The program reads like a research demonstration: it sets up gauge-field state, action terms, and stability checks, then prints a staged report with deterministic diagnostics. The target is not source recovery. It is an original implementation that reproduces the black-box contract.

## What to reproduce

- PROJECT 274 -- 计算凝聚态博士级科研合成项目
- 项目名称
- 电子-声子耦合与超导转变温度预测：高阶有限差分与稳定性分析（小规模可复现实验）
- > Electron-Phonon Coupling and Superconducting Transition Temperature Prediction:

## Command surface

```bash
./executable --help
./executable --version
./executable
```

A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

Expected `--version` text:

```text
synthesis-python-274 1.0
```

## Output cues

```text
PROJECT 274 -- Electron-Phonon Coupling and Tc Prediction
High-order finite differences, stability analysis,
small-scale reproducible experiment.
Pipeline execution
[pipeline] Stage 1 : adaptive k-point mesh        ... OK  (0.198s)
[pipeline] Stage 2 : Brillouin zone               ... OK  (0.003s)
[pipeline] Stage 3 : phonon-shell spectrum        ... OK  (0.003s)
[pipeline] Stage 4 : Eliashberg kernel            ... OK  (0.025s)
[pipeline] Stage 5 : spectral gap expansion       ... OK  (0.003s)
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make the no-argument path finish quickly under verifier time limits.
