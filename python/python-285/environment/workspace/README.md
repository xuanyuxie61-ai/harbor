# 多铁性材料磁电耦合模拟

The benchmark centers on a scientific driver in computational plasma physics, with behavior exposed through a local binary.

The program reads like a research demonstration: it sets up grid setup, conservative variables, and mode-growth summaries, then prints a staged report with deterministic diagnostics. The target is not source recovery. It is an original implementation that reproduces the black-box contract.

## Visible purpose

- PROJECT 285: 多铁性材料磁电耦合模拟 —— 高阶有限差分与稳定性分析
- > **计算材料前沿课题** | **博士级数值实验** | **15 个种子项目深度融合
- 一、科学问题与项目定位
- 1.1 核心科学问题

## Black-box probes

```bash
./executable --help
./executable --version
./executable
```

A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

Reference version:

```text
synthesis-python-285 1.0
```

## Reference cues

```text
PROJECT 285: 多铁性材料磁电耦合模拟
高阶有限差分与稳定性分析 (小规模可复现实验)
材料体系: BiFeO₃ (BFO) 钙钛矿多铁性材料
理论框架: Landau-Ginzburg-Devonshire 自由能泛函
数值方法: 6阶有限差分 + 半隐式时间积分 + ADI 分解
1. 数值方法验证
[1.1] 高阶有限差分精度测试
一阶导数误差:
O(h²)  : 4.0246e-02
```

## Rebuild target

Create your own Python implementation and a build script that produces `./executable` in the workspace root. Make the no-argument path finish quickly under verifier time limits.
