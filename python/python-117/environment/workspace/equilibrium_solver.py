"""
equilibrium_solver.py
=====================
平衡构型求解模块（源自 seed 225_cpr 的 Chebyshev Proxy Rootfinder）

在纳米颗粒-生物膜系统中，平衡距离 z_eq 定义为总相互作用力为零的位置：

    F_total(z) = F_elec(z) + F_vdw(z) + F_bend(z) + F_bind(z) = 0

该方程通常具有多个根（对应稳定/不稳定平衡态），且函数形式复杂（含指数、
幂律和样条插值）。本模块采用 **Chebyshev Proxy Rootfinder (CPR)** 算法，
通过 Chebyshev 多项式插值将求根问题转化为伴随矩阵特征值问题。

核心算法（源自 seed 225_cpr）：
    1. 在区间 [a, b] 上取 N+1 个 Chebyshev 节点（极值点）：
           x_k = cos(k*pi/N),  k=0..N
    2. 计算目标函数在这些节点上的样本 f_k = f(x_k)；
    3. 通过离散余弦变换（Clenshaw-Curtis 权重）得到 Chebyshev 展开系数 a_j；
    4. 尾部截断：从末尾开始累加系数绝对值，当累计值低于 epscutoff * max|a| 时截断；
    5. 构造 Chebyshev 伴随矩阵 A（基于第一类 Chebyshev 多项式的三递推关系）：
           A[0,1] = 1
           A[j,j-1] = 0.5, A[j,j+1] = 0.5  for j=1..Nt-2
           A[Nt-1, :] = -a[0:Nt] / (2*a[Nt])
           A[Nt-1, Nt-2] += 0.5
    6. 求 A 的特征值，其即为 Chebyshev 多项式的根；
    7. 筛选实根（|Im| < tau*|Re|）且落在 [-1,1] 内的特征值；
    8. 映射回 [a,b] 并排序。

数学公式：
    Chebyshev 多项式：T_n(cos theta) = cos(n*theta)
    插值展开：p(x) = sum_{j=0}^{N} a_j * T_j( (2x-a-b)/(b-a) )
    Clenshaw-Curtis 系数：
        a_j = (2/(N*p_j)) * sum_{k=0}^{N} (f_k * cos(j*k*pi/N) / p_k)
        其中 p_0 = p_N = 2, p_k = 1 (k=1..N-1)
"""

import numpy as np
from typing import List, Tuple, Callable


def chebyshev_coefficients_cpr(f_vals: np.ndarray, N: int) -> np.ndarray:
    """
    使用 Clenshaw-Curtis 公式计算 Chebyshev 展开系数（源自 seed 225_cpr）。

    Parameters
    ----------
    f_vals : ndarray, shape (N+1,)
        在 Chebyshev 极值点 x_k = cos(k*pi/N) 上的函数值。
    N : int
        Chebyshev 展开阶数。

    Returns
    -------
    acoeff : ndarray, shape (N+1,)
        Chebyshev 系数。
    """
    k = np.arange(N + 1)
    t = k * np.pi / N
    # 权重 p_j: p_0 = p_N = 2, 其余为 1
    pj = np.ones(N + 1, dtype=np.float64)
    pj[0] = 2.0
    pj[N] = 2.0
    acoeff = np.zeros(N + 1, dtype=np.float64)
    for j in range(N + 1):
        acoeff[j] = np.sum(f_vals * np.cos(j * t) / pj) * (2.0 / (N * pj[j]))
    return acoeff


def chebyshev_companion_matrix_cpr(acoeff: np.ndarray, Nt: int) -> np.ndarray:
    """
    构造 Chebyshev 伴随矩阵（源自 seed 225_cpr 核心算法）。

    Parameters
    ----------
    acoeff : ndarray
        Chebyshev 系数。
    Nt : int
        截断后的阶数（矩阵维度为 Nt x Nt）。

    Returns
    -------
    A : ndarray, shape (Nt, Nt)
        伴随矩阵。
    """
    A = np.zeros((Nt, Nt), dtype=np.float64)
    if Nt > 1:
        A[0, 1] = 1.0
        for j in range(1, Nt - 1):
            A[j, j - 1] = 0.5
            A[j, j + 1] = 0.5
        A[Nt - 1, :Nt] = -acoeff[:Nt] / (2.0 * acoeff[Nt])
        A[Nt - 1, Nt - 2] = A[Nt - 1, Nt - 2] + 0.5
    else:
        A[0, 0] = -acoeff[0] / (2.0 * acoeff[1])
    return A


def chebyshev_proxy_rootfinder(f: Callable[[float], float],
                               a: float,
                               b: float,
                               N: int = 64,
                               epscutoff: float = 1e-13,
                               tau: float = 1e-8,
                               sigma: float = 1e-6) -> List[float]:
    """
    Chebyshev Proxy Rootfinder（源自 seed 225_cpr 核心算法）。

    Parameters
    ----------
    f : callable
        目标函数 f(x)，定义在 [a,b] 上。
    a, b : float
        搜索区间。
    N : int
        Chebyshev 插值阶数。
    epscutoff : float
        系数截断相对阈值。
    tau : float
        复根的虚部相对容差。
    sigma : float
        区间外根容差。

    Returns
    -------
    roots : list of float
        按升序排列的实根。
    """
    # [HOLE 3] 请补全 Chebyshev Proxy Rootfinder 核心算法：
    # 1. 在 [-1,1] 上取 N+1 个 Chebyshev 极值点 xi = cos(k*pi/N)，并映射到 [a,b]
    # 2. 计算目标函数在这些节点上的样本 f_nodes
    # 3. 调用 chebyshev_coefficients_cpr(f_nodes, N) 计算 Chebyshev 系数
    # 4. 对系数进行尾部截断（从末尾累加，当 tailnorm < epscutoff * max|a| 时截断）
    # 5. 调用 chebyshev_companion_matrix_cpr(acoeff, Nt) 构造伴随矩阵
    # 6. 计算伴随矩阵的特征值
    # 7. 筛选：保留 |Im| < tau*|Re| 且 |Re| <= 1+sigma 的特征值，映射回 [a,b]
    # 8. 去重并排序后返回
    # TODO: 实现上述 CPR 算法流程
    raise NotImplementedError("HOLE 3: 请补全 chebyshev_proxy_rootfinder 的 CPR 算法")


def find_equilibrium_distances(force_func: Callable[[float], float],
                                z_min: float = 0.1,
                                z_max: float = 10.0) -> Tuple[List[float], List[str]]:
    """
    寻找总作用力为零的平衡距离，并判断稳定性。

    稳定性判据：F'(z_eq) < 0 为稳定（回复力），F'(z_eq) > 0 为不稳定。

    Parameters
    ----------
    force_func : callable
        总力函数 F(z)。
    z_min, z_max : float
        搜索区间。

    Returns
    -------
    roots : list
        平衡距离列表。
    stability : list
        对应的稳定性标签（'stable' / 'unstable'）。
    """
    roots = chebyshev_proxy_rootfinder(force_func, z_min, z_max, N=80)
    stability = []
    h = 1e-4
    for z_eq in roots:
        df = (force_func(z_eq + h) - force_func(z_eq - h)) / (2.0 * h)
        if df < 0:
            stability.append("stable")
        else:
            stability.append("unstable")
    return roots, stability
