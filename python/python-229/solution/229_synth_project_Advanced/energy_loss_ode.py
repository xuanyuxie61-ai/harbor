"""
energy_loss_ode.py
==================

来源: 488_grazing_ode + 880_polar_ode
-------------------------------------
- 488_grazing_ode:  植物-食草动物非线性耦合 ODE, 具有饱和型相互作用项
                      du/dt = r1 u (1 - u/k) - c1 v (1 - exp(-d1 u))
                      dv/dt = -a v + c2 v (1 - exp(-d2 u))
- 880_polar_ode:    复平面极坐标 ODE 基准, r(θ) = 1 - sin(θ) cos(3θ),
                      解析解 z(t) = r(t) exp(i t)

物理重构
--------
我们将 grazing ODE 的形式重构为 **带电粒子在介质中的连续慢化 + 离散
辐射涨落** 模型:

    dE/dx = S_coll(E) + S_rad(E) + ξ_stoch(E, x)

其中:
    S_coll(E)  = - (α E) (1 - E/E_max)             碰撞能量损失 (Bethe-Bloch 唯象近似)
    S_rad(E)   = - (β E^2) / E_c                    辐射损失 (bremsstrahlung, 正比 E^2)
    ξ_stoch    = -γ E (1 - exp(-δ E)) · n(x)        离散 Landau 涨落, n(x) 为随机散射中心密度

这与 grazing 模型的结构同构:
    u ↔ E (能量),  v ↔ n (散射中心密度)
    r1 u(1 - u/k) ↔ α E (1 - E/E_max)    (自限制增长/损失)
    c v (1 - exp(-d u)) ↔ γ E (1 - exp(-δ E)) n(x)   (饱和型相互作用)

极坐标 ODE 基准用于验证数值积分精度, 其解析解为:
    z_exact(t) = r(t) exp(i t),  r(t) = 1 - sin(t) cos(3t)

数值求解器
----------
自适应步长 RK45 (Dormand-Prince), 带事件检测 (E → 0 终止).
"""

from __future__ import annotations
from typing import Callable, Tuple, List
import math


# ===========================================================================
#                  Grazing 型能量损失 ODE
# ===========================================================================

class EnergyLossParams:
    """
    能量损失 ODE 参数.

    物理量纲:
        E  : GeV
        x  : 辐射长度 X_0 (无量纲, 介质穿透深度 / X_0)
        α  : 碰撞损失系数 (1/X_0), 典型 ~0.02
        E_max : 运动学上限 (入射能量, GeV)
        β  : 辐射损失系数 / 临界能量 E_c, 典型 ~1e-3
        γ  : 随机涨落强度
        δ  : 涨落饱和标度 (1/GeV)
        n0 : 平均散射中心密度
    """

    def __init__(
        self,
        alpha: float = 0.02,
        E_max: float = 100.0,
        beta: float = 1.0e-3,
        E_c: float = 10.0,
        gamma: float = 0.01,
        delta: float = 0.1,
        n0: float = 1.0,
    ):
        if alpha < 0 or E_max <= 0 or beta < 0 or E_c <= 0:
            raise ValueError("EnergyLossParams: 参数必须非负/正")
        self.alpha = alpha
        self.E_max = E_max
        self.beta = beta
        self.E_c = E_c
        self.gamma = gamma
        self.delta = delta
        self.n0 = n0


def grazing_energy_loss_rhs(
    y: Tuple[float, float],
    params: EnergyLossParams,
) -> Tuple[float, float]:
    """
    右端函数: (E, n) → (dE/dx, dn/dx).

    状态变量:
        y[0] = E  : 粒子能量 (GeV)
        y[1] = n  : 局部散射中心密度 (归一化)

    方程:
        dE/dx = -α E (1 - E/E_max) - β E^2 / E_c - γ E (1 - exp(-δ E)) n
        dn/dx = -a n + c n (1 - exp(-d E))    (密度演化, 类比 grazing v 方程)

    边界:
        E = 0 ⇒ 停止;  n < 0 ⇒ 截断为 0
    """
    E, n = y
    p = params
    # 安全: E 必须非负
    E = max(E, 0.0)
    n = max(n, 0.0)

    # 碰撞损失 (Bethe-Bloch 唯象)
    coll = -p.alpha * E * (1.0 - E / p.E_max)
    # 辐射损失 (bremsstrahlung)
    rad = -p.beta * E * E / p.E_c
    # 随机涨落 (饱和型)
    stoch = -p.gamma * E * (1.0 - math.exp(-p.delta * E)) * n

    dEdx = coll + rad + stoch

    # 密度演化 (grazing v 方程)
    a_coeff = 0.1
    c_coeff = 0.2
    d_coeff = 0.05
    dndx = -a_coeff * n + c_coeff * n * (1.0 - math.exp(-d_coeff * E))

    return (dEdx, dndx)


# ===========================================================================
#            极坐标 ODE 基准 (来自 880_polar_ode)
# ===========================================================================

def polar_exact(t: float) -> complex:
    """
    解析解 z(t) = r(t) exp(i t),  r(t) = 1 - sin(t) cos(3t).

    用于验证 ODE 求解器的收敛阶.
    """
    r = 1.0 - math.sin(t) * math.cos(3.0 * t)
    return r * complex(math.cos(t), math.sin(t))


def polar_rhs(t: float, z: complex) -> complex:
    """
    dz/dt 的右端函数.

    z = r exp(i θ), θ = t
    dz/dt = (dr/dt + i r) exp(i t)
    dr/dt = -cos(t) cos(3t) + 3 sin(t) sin(3t)
    """
    r = 1.0 - math.sin(t) * math.cos(3.0 * t)
    drdt = -math.cos(t) * math.cos(3.0 * t) + 3.0 * math.sin(t) * math.sin(3.0 * t)
    exp_it = complex(math.cos(t), math.sin(t))
    return (drdt + 1j * r) * exp_it


# ===========================================================================
#            自适应 RK45 (Dormand-Prince) 求解器
# ===========================================================================

# Dormand-Prince 系数
_DP_A = [
    [],
    [1.0 / 5],
    [3.0 / 40, 9.0 / 40],
    [44.0 / 45, -56.0 / 15, 32.0 / 9],
    [19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729],
    [9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656],
    [35.0 / 384, 0.0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84],
]
_DP_B5 = [35.0 / 384, 0.0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0.0]
_DP_B4 = [
    5179.0 / 57600, 0.0, 7571.0 / 16695, 393.0 / 640,
    -92097.0 / 339200, 187.0 / 2100, 1.0 / 40,
]
_DP_C = [0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0, 1.0]


def rk45_step(
    f: Callable,
    t: float,
    y,
    h: float,
):
    """
    单步 Dormand-Prince, 返回 (y5, y4, err).
    y 可以是 float / complex / tuple.
    """
    if isinstance(y, tuple):
        k = [None] * 7
        k[0] = f(t, y) if callable_with_t(f) else f(y)
        for s in range(1, 7):
            ys = tuple(
                y[i] + h * sum(_DP_A[s][j] * k[j][i] for j in range(s))
                for i in range(len(y))
            )
            k[s] = f(t + _DP_C[s] * h, ys) if callable_with_t(f) else f(ys)
        y5 = tuple(
            y[i] + h * sum(_DP_B5[s] * k[s][i] for s in range(7))
            for i in range(len(y))
        )
        y4 = tuple(
            y[i] + h * sum(_DP_B4[s] * k[s][i] for s in range(7))
            for i in range(len(y))
        )
        err = max(abs(y5[i] - y4[i]) for i in range(len(y)))
        return y5, y4, err
    # 标量或复数
    k = [None] * 7
    k[0] = f(t, y) if callable_with_t(f) else f(y)
    for s in range(1, 7):
        ys = y + h * sum(_DP_A[s][j] * k[j] for j in range(s))
        k[s] = f(t + _DP_C[s] * h, ys) if callable_with_t(f) else f(ys)
    y5 = y + h * sum(_DP_B5[s] * k[s] for s in range(7))
    y4 = y + h * sum(_DP_B4[s] * k[s] for s in range(7))
    err = abs(y5 - y4)
    return y5, y4, err


def callable_with_t(f) -> bool:
    """探测 f 是否接受两个参数 (t, y)."""
    try:
        import inspect
        sig = inspect.signature(f)
        return len(sig.parameters) >= 2
    except Exception:
        return True


def rk45_adaptive(
    f,
    t0: float,
    y0,
    t_end: float,
    h0: float = 1e-3,
    tol: float = 1e-8,
    h_min: float = 1e-12,
    h_max: float = 1.0,
    event=None,
) -> Tuple[List[float], List]:
    """
    自适应 RK45 积分.

    event: 可选事件函数 event(y) → float, 过零时停止.

    返回 (t_list, y_list).
    """
    t = t0
    y = y0
    h = h0
    t_list = [t]
    y_list = [y]

    safety = 0.9
    max_steps = 100000

    for _ in range(max_steps):
        if (h > 0 and t >= t_end) or (h < 0 and t <= t_end):
            break
        # 防止越过 t_end
        if h > 0 and t + h > t_end:
            h = t_end - t
        elif h < 0 and t + h < t_end:
            h = t_end - t

        if callable_with_t(f):
            y5, y4, err = rk45_step(lambda yy: f(t, yy), t, y, h)
        else:
            y5, y4, err = rk45_step(f, t, y, h)

        if err < 1e-300:
            err = 1e-300
        # PI 步长控制
        if err <= tol:
            t += h
            y = y5
            t_list.append(t)
            y_list.append(y)
            if event is not None:
                try:
                    if event(y) <= 0:
                        break
                except Exception:
                    pass
        # 更新步长
        factor = safety * (tol / err) ** 0.2
        factor = max(0.2, min(5.0, factor))
        h = h * factor
        h = max(h_min, min(h_max, abs(h))) * (1.0 if h > 0 else -1.0)

    return t_list, y_list


# ===========================================================================
#            能量损失轨迹计算
# ===========================================================================

def compute_energy_loss_trajectory(
    E0: float,
    n0: float = 1.0,
    x_max: float = 10.0,
    params: EnergyLossParams = None,
) -> Tuple[List[float], List[float]]:
    """
    计算能量随穿透深度 x 的演化.

    返回 (x_list, E_list).
    """
    if params is None:
        params = EnergyLossParams()
    if E0 <= 0:
        raise ValueError("E0 必须 > 0")

    def rhs(y):
        return grazing_energy_loss_rhs(y, params)

    def event(y):
        return y[0] - 1e-6  # 当 E < 1e-6 时停止

    xs, ys = rk45_adaptive(
        rhs, 0.0, (E0, n0), x_max,
        h0=0.01, tol=1e-9, event=event,
    )
    E_list = [max(y[0], 0.0) for y in ys]
    return xs, E_list


def polar_ode_benchmark(
    t_end: float = 2.0 * math.pi,
    n_steps: int = 100,
) -> Tuple[List[float], List[complex], List[complex]]:
    """
    极坐标 ODE 基准测试.

    返回 (t, z_numeric, z_exact).
    """
    def rhs(t, z):
        return polar_rhs(t, z)

    ts, zs = rk45_adaptive(
        rhs, 0.0, 0j, t_end,
        h0=t_end / n_steps, tol=1e-10, h_max=t_end / 10,
    )
    z_exact = [polar_exact(t) for t in ts]
    return ts, zs, z_exact


__all__ = [
    "EnergyLossParams",
    "grazing_energy_loss_rhs",
    "polar_exact", "polar_rhs",
    "rk45_step", "rk45_adaptive",
    "compute_energy_loss_trajectory",
    "polar_ode_benchmark",
]
