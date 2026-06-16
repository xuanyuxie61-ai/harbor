# Reverse-Engineer Project 202: 不确定性量化: 随机配置方法 (Stochastic Collocation UQ)

## Build goal

You are in `/app/workspace` with a reference executable for project `202`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

The reference binary reports the following version for project 202:

```text
synthesis-python-202 1.0
```

This case exposes the observable surface of a research-style uncertainty quantification driver rather than its source. Probe conservatively; the hidden verifier focuses on observable behavior rather than internal file names.

## What must match

Project 202 should be rebuilt around:

```text
不确定性量化: 随机配置方法 (Stochastic Collocation UQ)
```

Useful observation anchors from the oracle:

```text
不确定性量化: 随机配置方法 (Stochastic Collocation UQ)
PROJECT 202 — 博士级科研代码合成项目
NumPy 版本: 1.26.4
随机种子: 42 (可复现)
第一部分: Smolyak 稀疏网格构造
维度 D=3, Level q=4
```

Finish by making a standalone workspace executable for 不确定性量化: 随机配置方法 (Stochastic Collocation UQ); hidden tests will run the build script before probing it.

Treat the oracle as read-only evidence for project 202, not as a runtime dependency. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
