# Reverse-Engineer Project 252: 黑洞喷流形成与 MHD 模拟

## Reconstruction goal

You are in `/app/workspace` with a reference executable for project `252`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The observed version banner for 黑洞喷流形成与 MHD 模拟 is:

```text
synthesis-python-252 1.0
```

The benchmark centers on a scientific driver in computational plasma physics, with behavior exposed through a local binary. A good reconstruction usually comes from comparing the short flag outputs with several captures of the default run.

## Observable behavior

Reconstruct the command-line behavior for this computational plasma physics target:

```text
黑洞喷流形成与 MHD 模拟
```

Start by matching these lines:

```text
PROJECT 252: 黑洞喷流形成与 MHD 模拟
高阶有限差分与稳定性分析 (博士级可复现实验)
黑洞喷流 MHD 模拟 & 稳定性分析 (博士级可复现实验)
完成 20 步, 最终密度扰动: 7.277096e-02
[1/8] 配置校验: 全部通过 (9 项)
[2/8] 网格生成: Nr=48, Nt=24, r=[2.00, 50.0]
```

Do the reconstruction in source form, then make `compile.sh` or `build.sh` produce the executable file expected by Harbor.

Keep project 252 cleanroom: no delegation to the reference executable and no original-repo download. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
