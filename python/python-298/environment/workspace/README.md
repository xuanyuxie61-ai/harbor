# 高阶有限差分 PIC 稳定性分析

The visible binary represents a synthesized computational plasma physics workflow with a deterministic command-line transcript.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. The hidden tests may look beyond the examples, so preserve the general report shape rather than only one line.

## What to reproduce

- PROJECT 298: 计算等离子体 PIC 方法高阶有限差分稳定性分析
- 统一入口: 零参数运行
- 科学问题:
- 研究一维静电 PIC 模拟中, 高阶有限差分格式 (2/4/6/8 阶) 对数值色散、

## Command surface

```bash
./executable --help
./executable --version
./executable
```

Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

Expected `--version` text:

```text
synthesis-python-298 1.0
```

## Output cues

```text
PROJECT 298: 高阶有限差分 PIC 稳定性分析
电子数密度 n_e        = 1.000e+20 m^-3
电子温度 T_e          = 100.00 eV (1.160e+06 K)
等离子体频率 omega_pe = 5.641e+11 rad/s
[PART 1] 等离子体物理参数
德拜长度 lambda_D     = 7.434e-06 m
热速度 v_th           = 5.931e+06 m/s
等离子体参数 Lambda   = 1.721e+05
库仑对数 ln_Lambda    = 13.790
```

## Expected deliverable

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.
