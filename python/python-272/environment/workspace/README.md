# Weyl 半金属 Berry Curvature 计算系统

This is a ProgramBench-style task: infer and reproduce a spectral physics CLI from documentation plus black-box runs.

The program reads like a research demonstration: it sets up operators, spectra, and formatted numerical landmarks, then prints a staged report with deterministic diagnostics. The verifier rewards the public interface: executable creation, flags, deterministic output, and selected anchors.

## Scientific role

- Weyl 半金属 Berry Curvature 计算：高阶有限差分与稳定性分析
- 主入口文件
- 科学问题：
- Weyl 半金属是一类具有非平庸拓扑性质的新型量子材料。其低能激发

## Useful probes

```bash
./executable --help
./executable --version
./executable
```

Start with the version and help flags, then run the binary without arguments to collect the full staged report.

Stable version text:

```text
synthesis-python-272 1.0
```

## Report signatures

```text
Weyl 半金属 Berry Curvature 计算系统
高阶有限差分与稳定性分析 (小规模可复现实验)
第 1 部分：Weyl Hamiltonian 构建与能带结构
本征值: E_± = -0.2450, 0.3550
[Hamiltonian] Γ 点 Hamiltonian 矩阵:
H(Γ) =
[[+0.0550 -0.3000 + +0.0000i],
[-0.3000 +0.0550 + +0.0000i]]
[Hamiltonian] Γ 点能隙: Δ = 0.6000
```

## Workspace contract

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep the generated executable at the workspace root and make it executable.
