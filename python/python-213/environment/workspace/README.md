# 反应-扩散系统的 PDE 约束最优控制

This case exposes the observable surface of a research-style fusion and radiation transport driver rather than its source.

The program reads like a research demonstration: it sets up mesh data, cross sections, and solver diagnostics, then prints a staged report with deterministic diagnostics. Avoid delegating to the supplied executable; tests remove unchanged reference binaries before building.

## Visible purpose

- 反应-扩散系统的 PDE 约束最优控制
- 博士级科学计算合成项目
- Python 版本: 3.11.11
- NumPy 版本: 1.26.4

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

Reference version:

```text
synthesis-python-213 1.0
```

## Reference cues

```text
PROJECT 213: 凸优化与内点法
反应-扩散系统的 PDE 约束最优控制
博士级科学计算合成项目
Python 版本: 3.11.11
NumPy 版本: 1.26.4
演示 1: 凸多边形优化域 (882_polygon)
域顶点数: 6
凸性检测: 凸
面积: 2.457347
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make `compile.sh` idempotent; the verifier may run it in a workspace that already contains files.
