"""
pbpk_stochastic.py
基于种子项目 424_feynman_kac_3d

实现 Feynman-Kac 公式驱动的 3D 随机游走模拟，
用于求解椭圆型边界值问题：
    (1/2) ΔU - V(x) U = 0   in Ω
    U = g(x)                on ∂Ω

在 PBPK 模型中用于：
- 模拟药物分子在组织内的随机布朗运动与边界吸收
- Monte Carlo 估计药物在器官内的稳态空间分布
- 评估药物到达靶点细胞的概率（ hitting probability ）
"""

import numpy as np
from typing import Callable, Tuple

# ---------------------------------------------------------------------------
# 3D 随机游走与 Feynman-Kac 估计
# ---------------------------------------------------------------------------

def random_walk_3d_step(pos: np.ndarray, h: float) -> np.ndarray:
    """
    执行一次 3D 离散随机游走步进。
    从 6 个方向中均匀随机选择一个方向，步长为 sqrt(3*h)。
    """
    direction = np.random.randint(0, 6)
    step = np.zeros(3)
    step_size = np.sqrt(3.0 * h)
    if direction == 0:
        step[0] = step_size
    elif direction == 1:
        step[0] = -step_size
    elif direction == 2:
        step[1] = step_size
    elif direction == 3:
        step[1] = -step_size
    elif direction == 4:
        step[2] = step_size
    else:
        step[2] = -step_size
    return pos + step


def inside_ellipsoid(pos: np.ndarray, a: float, b: float, c: float) -> bool:
    """
    判断点是否在椭球内：(x/a)^2 + (y/b)^2 + (z/c)^2 <= 1。
    """
    val = (pos[0] / a) ** 2 + (pos[1] / b) ** 2 + (pos[2] / c) ** 2
    return val <= 1.0


def feynman_kac_3d_monte_carlo(x0: float, y0: float, z0: float,
                                a: float, b: float, c: float,
                                potential: Callable[[np.ndarray], float],
                                boundary_value: Callable[[np.ndarray], float],
                                h: float = 0.01, n_trajectories: int = 10000,
                                max_steps: int = 100000) -> Tuple[float, float]:
    """
    使用 Feynman-Kac 公式的 Monte Carlo 估计求解 3D 椭圆 BVP：
        U(x0) ≈ E[ exp(-∫_0^τ V(X_s) ds) * g(X_τ) ]
    其中 τ 为首次退出时间。

    参数：
        x0,y0,z0 : 初始位置（必须在椭球内）
        a,b,c    : 椭球半轴
        potential: 势能函数 V(x)
        boundary_value: 边界值函数 g(x)
        h        : 时间步长
        n_trajectories: 轨迹数
        max_steps: 单条轨迹最大步数
    返回：
        (mean, std_error)
    """
    pos0 = np.array([x0, y0, z0])
    if not inside_ellipsoid(pos0, a, b, c):
        raise ValueError("Initial position must be inside the ellipsoid")
    if h <= 0.0 or n_trajectories <= 0:
        raise ValueError("Invalid Monte Carlo parameters")

    estimates = np.empty(n_trajectories)
    for k in range(n_trajectories):
        pos = pos0.copy()
        Y = 1.0  # Feynman-Kac 路径泛函
        steps = 0
        while steps < max_steps:
            V = potential(pos)
            # 显式 Euler 更新泛函
            Y *= np.exp(-V * h)
            pos = random_walk_3d_step(pos, h)
            steps += 1
            if not inside_ellipsoid(pos, a, b, c):
                break
        # 边界值
        g = boundary_value(pos)
        estimates[k] = Y * g

    mean = np.mean(estimates)
    se = np.std(estimates, ddof=1) / np.sqrt(n_trajectories)
    return mean, se


# ---------------------------------------------------------------------------
# PBPK 组织吸收概率
# ---------------------------------------------------------------------------

def drug_absorption_probability_organ(center: np.ndarray, organ_axes: np.ndarray,
                                       D_eff: float, clearance: float,
                                       n_trajectories: int = 5000) -> Tuple[float, float]:
    """
    计算药物分子从血管入口到达器官内部某点的吸收保留概率。
    使用 Feynman-Kac 公式，将清除率视为势能项 V = clearance/D_eff。
    """
    if len(center) != 3 or len(organ_axes) != 3:
        raise ValueError("center and organ_axes must be 3D vectors")
    a, b, c = organ_axes
    x0, y0, z0 = center

    def potential(pos):
        return clearance / max(D_eff, 1e-20)

    def boundary_value(pos):
        # 边界上药物被血流带走，浓度归一化为 0
        return 0.0

    mean, se = feynman_kac_3d_monte_carlo(x0, y0, z0, a, b, c,
                                           potential, boundary_value,
                                           h=0.001, n_trajectories=n_trajectories,
                                           max_steps=50000)
    return mean, se


def organ_hitting_probability(source_pos: np.ndarray, target_pos: np.ndarray,
                               organ_axes: np.ndarray, D_eff: float,
                               n_trajectories: int = 2000,
                               h: float = 1e-6) -> float:
    """
    计算药物分子从 source_pos 出发，在到达器官边界前 hit 到 target_pos
    附近小球的概率。用于评估靶向给药效率。
    """
    if len(source_pos) != 3 or len(target_pos) != 3:
        raise ValueError("Positions must be 3D")
    a, b, c = organ_axes
    if not inside_ellipsoid(source_pos, a, b, c):
        raise ValueError("Source must be inside organ")
    if not inside_ellipsoid(target_pos, a, b, c):
        raise ValueError("Target must be inside organ")

    epsilon = 0.05 * min(a, b, c)  # hit 球半径
    hits = 0
    max_steps = 100000
    for _ in range(n_trajectories):
        pos = source_pos.copy()
        for _ in range(max_steps):
            pos = random_walk_3d_step(pos, h)
            dist = np.linalg.norm(pos - target_pos)
            if dist < epsilon:
                hits += 1
                break
            if not inside_ellipsoid(pos, a, b, c):
                break
    return hits / n_trajectories


# ---------------------------------------------------------------------------
# 一维随机游走的 Feynman-Kac（快速验证）
# ---------------------------------------------------------------------------

def feynman_kac_1d(x0: float, L: float, V_const: float,
                    g_left: float, g_right: float,
                    h: float = 0.001, n_trajectories: int = 5000) -> float:
    """
    一维 Feynman-Kac：求解 U''/2 - V U = 0, U(0)=g_left, U(L)=g_right。
    解析解（V=const>0）：
        U(x) = A sinh(√(2V) x) + B cosh(√(2V) x)
    其中 A,B 由边界条件确定。
    """
    if not (0.0 <= x0 <= L):
        raise ValueError("x0 must be in [0, L]")
    estimates = np.empty(n_trajectories)
    for k in range(n_trajectories):
        x = x0
        Y = 1.0
        while True:
            Y *= np.exp(-V_const * h)
            # 1D 步进：±sqrt(h)
            x += np.sqrt(h) if np.random.rand() < 0.5 else -np.sqrt(h)
            if x <= 0.0:
                estimates[k] = Y * g_left
                break
            if x >= L:
                estimates[k] = Y * g_right
                break
    return np.mean(estimates)


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 测试 1D Feynman-Kac
    L, V, gL, gR = 1.0, 2.0, 0.0, 1.0
    x0 = 0.5
    mc_val = feynman_kac_1d(x0, L, V, gL, gR, h=0.0005, n_trajectories=20000)
    # 解析解
    sqrt2V = np.sqrt(2.0 * V)
    A = (gR - gL * np.cosh(sqrt2V * L)) / np.sinh(sqrt2V * L)
    B = gL
    exact = A * np.sinh(sqrt2V * x0) + B * np.cosh(sqrt2V * x0)
    print(f"1D Feynman-Kac: MC={mc_val:.6f}, Exact={exact:.6f}, RelErr={abs(mc_val-exact)/exact:.4e}")
    # 测试 3D
    mean, se = feynman_kac_3d_monte_carlo(0.0, 0.0, 0.0, 1.0, 0.8, 0.6,
                                           lambda p: 1.0,
                                           lambda p: np.exp((p[0]/1.0)**2 + (p[1]/0.8)**2 + (p[2]/0.6)**2 - 1.0),
                                           h=0.01, n_trajectories=2000)
    print(f"3D Feynman-Kac estimate: {mean:.6f} ± {se:.6f}")
