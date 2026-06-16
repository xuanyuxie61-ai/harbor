# Reverse-Engineer Project 215: 设计一维催化反应器的密度分布 ρ(x) ∈ [0,1]^n,

## Cleanroom objective

You are in `/app/workspace` with a reference executable for project `215`. Rebuild the program as original Python code that creates a compatible `/app/workspace/executable`.

Project 215's identity check should return:

```text
synthesis-python-215 1.0
```

Use this task as a cleanroom target for reconstructing a scientific Python command-line workflow in scientific computing. The binary is meant for observation only; rebuild the behavior in your own files after probing it.

## Behavioral target

The public-facing subject of project 215 is:

```text
设计一维催化反应器的密度分布 ρ(x) ∈ [0,1]^n,
```

Visible cues for the scientific computing workflow:

```text
f1: 结构柔度    (最小化 → 最大化刚度)
f2: 热应力方差  (最小化 → 均匀温度分布)
f3: 负转化率    (最小化 → 最大化生化转化效率)
约束: 体积分数 ≤ 0.5
科学问题:
设计一维催化反应器的密度分布 ρ(x) ∈ [0,1]^n,
```

The final workspace for project 215 needs your implementation and a build script that regenerates `./executable`.

Do not copy, unpack, wrap, or call the supplied binary while solving 设计一维催化反应器的密度分布 ρ(x) ∈ [0,1]^n,; create a fresh implementation. Hidden tests may probe flags, exit codes, report order, and selected textual or numeric anchors beyond the examples.
