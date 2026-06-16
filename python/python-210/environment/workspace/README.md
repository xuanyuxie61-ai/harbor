# 不确定性量化 — 可靠性分析与失效概率计算

The visible binary represents a synthesized uncertainty quantification workflow with a deterministic command-line transcript.

The program reads like a research demonstration: it sets up statistical estimators, quadrature rules, and compact diagnostic reports, then prints a staged report with deterministic diagnostics. The hidden tests may look beyond the examples, so preserve the general report shape rather than only one line.

## Observed behavior

- Uncertainty Quantification: Reliability Analysis &
- Failure Probability Computation
- 问题定义
- 随机变量数        = 2

## CLI checks

```bash
./executable --help
./executable --version
./executable
```

Use the flags to confirm command handling, then mine the normal transcript for labels, dimensions, and numeric anchors.

Identity check:

```text
synthesis-python-210 1.0
```

## Transcript anchors

```text
PROJECT 210: 不确定性量化 — 可靠性分析与失效概率计算
Uncertainty Quantification: Reliability Analysis &
Failure Probability Computation
随机变量数        = 2
问题定义
X1_Strength         : mean = 1.0000, std = 0.2000
X2_Load             : mean = 0.0000, std = 1.0000
极限状态函数: 应力-强度干涉模型
g(u) = (mu_R + sigma_R*u_1) - (mu_S + sigma_S*u_2)
```

## Submission shape

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Keep constants close to the observed transcript, but avoid copying source files from outside the workspace.
