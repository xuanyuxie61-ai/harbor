"""
裂变势能面的 Chebyshev 级数逼近
=================================
融合原始项目:
  - 1271_toms446/cheby.m: Chebyshev 分析

科学背景:
---------
Chebyshev 多项式 T_n(x) 在区间 [-1,1] 上关于权函数 w(x)=(1-x²)^(-1/2) 正交:
  ∫_{-1}^{1} T_m(x) T_n(x) / √(1-x²) dx = π/2 δ_{mn} (n>0), π (n=0)

对于裂变势能面 V(β₂)，我们将其映射到标准区间后展开:
  V(x) ≈ Σ_{k=0}^{N-1} c_k T_k(x),   x = (β₂ - β_c) / β_w

Chebyshev 级数的系数通过离散正交性（Clenshaw-Curtis 节点）计算:
  x_j = cos(jπ / N),  j=0,...,N
  c_k = (2/N) Σ_{j=0}^{N}'' V(x_j) cos(π k j / N)
其中双撇号表示首尾项取半权重。

Chebyshev 逼近的优势:
1. 在最大模意义下接近最优多项式逼近（等振荡定理）
2. 系数 c_k 的快速衰减指示函数光滑性
3. 便于解析求导:
   V'(x) = Σ_{k=1}^{N-1} c_k k U_{k-1}(x)
   其中 U_n(x) 为第二类 Chebyshev 多项式

本模块还实现基于 Chebyshev 展开的力 F = -dV/dβ₂ 的快速计算。
"""

import numpy as np
from typing import Callable, Tuple


def chebyshev_nodes(n: int) -> np.ndarray:
    """
    生成 Chebyshev 第二类节点 (Clenshaw-Curtis):
    x_j = cos(jπ / n), j=0,...,n
    """
    if n < 1:
        raise ValueError("n must be positive")
    j = np.arange(n + 1)
    return np.cos(np.pi * j / n)


def chebyshev_coefficients(
    f: Callable[[np.ndarray], np.ndarray],
    n: int,
    a: float = -1.0,
    b: float = 1.0,
) -> np.ndarray:
    """
    计算函数 f 在 [a,b] 上的 n 阶 Chebyshev 展开系数.
    
    改编自 cheby.m 的核心算法，使用 DCT-I 变换。
    
    参数:
        f: 目标函数，接受数组返回数组
        n: 展开阶数 (节点数 = n+1)
        a, b: 实际区间端点
    返回:
        系数数组 c[0..n]
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    
    # Clenshaw-Curtis 节点（标准区间 [-1,1]）
    x_std = chebyshev_nodes(n)
    
    # 映射到实际区间
    x_phys = 0.5 * (a + b) + 0.5 * (b - a) * x_std
    
    # 函数采样
    fx = f(x_phys)
    
    # DCT-I 计算系数
    # c_k = (2/n) * Σ_{j=0}^{n}'' fx_j * cos(π k j / n)
    c = np.zeros(n + 1)
    for k in range(n + 1):
        j = np.arange(n + 1)
        cos_terms = np.cos(np.pi * k * j / n)
        # 首尾半权重
        weights = np.ones(n + 1)
        weights[0] = 0.5
        weights[-1] = 0.5
        c[k] = (2.0 / n) * np.sum(weights * fx * cos_terms)
    
    return c


def chebyshev_evaluate(x: np.ndarray, c: np.ndarray, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """
    使用 Clenshaw 递推公式计算 Chebyshev 级数.
    
    对于标准变量 t ∈ [-1,1]，递推:
    b_N = c_N, b_{N+1} = 0
    b_k = 2t b_{k+1} - b_{k+2} + c_k,  k=N-1,...,0
    V(t) = (b_0 - b_2) / 2
    
    参数:
        x: 物理坐标数组
        c: Chebyshev 系数
        a, b: 物理区间
    返回:
        级数值
    """
    # 映射到标准区间
    t = (2.0 * x - (a + b)) / (b - a)
    t = np.clip(t, -1.0, 1.0)
    
    N = len(c) - 1
    
    # Clenshaw 递推（向量化版）
    b_kp2 = np.zeros_like(t, dtype=float)
    b_kp1 = np.full_like(t, c[N])
    
    for k in range(N - 1, 0, -1):
        b_k = 2.0 * t * b_kp1 - b_kp2 + c[k]
        b_kp2 = b_kp1
        b_kp1 = b_k
    
    result = t * b_kp1 - b_kp2 + c[0]
    return result


def chebyshev_derivative_coefficients(c: np.ndarray) -> np.ndarray:
    """
    由 Chebyshev 系数 c_k 计算导数级数系数 d_k.
    
    递推关系:
    d_{N-1} = 2N c_N
    d_{N-2} = 2(N-1) c_{N-1}
    d_k = d_{k+2} + 2(k+1) c_{k+1},  k=N-3,...,0
    d_0 = d_2 / 2 + c_1
    
    注意这是标准区间 [-1,1] 上的导数，需要再乘以 2/(b-a) 得到物理导数。
    """
    N = len(c) - 1
    if N < 1:
        return np.zeros_like(c)
    
    d = np.zeros(N)
    d[N - 1] = 2.0 * N * c[N]
    if N >= 2:
        d[N - 2] = 2.0 * (N - 1) * c[N - 1]
    
    for k in range(N - 3, -1, -1):
        d[k] = d[k + 2] + 2.0 * (k + 1) * c[k + 1]
    
    # 修正 d_0
    # d[0] 已经是正确的，因为递推公式在 k=0 时给出 d_0 = d_2 + 2c_1
    # 但标准公式中导数展开的首项系数是 d_0，而 T_0=1，不需要额外处理
    return d


def chebyshev_derivative(
    x: np.ndarray,
    c: np.ndarray,
    a: float = -1.0,
    b: float = 1.0,
) -> np.ndarray:
    """
    计算 Chebyshev 逼近函数的物理导数 dV/dx.
    """
    d = chebyshev_derivative_coefficients(c)
    # 标准区间导数
    dV_dt = chebyshev_evaluate(x, np.append(d, 0.0), a, b)
    # 链式法则: dV/dx = dV/dt * dt/dx = dV/dt * 2/(b-a)
    dV_dx = dV_dt * 2.0 / (b - a)
    return dV_dx


def build_chebyshev_pes_approximation(
    mass_number: int,
    charge_number: int,
    beta2_min: float = -0.3,
    beta2_max: float = 2.5,
    n_terms: int = 32,
) -> Tuple[np.ndarray, float, float]:
    """
    为裂变势能面构建 Chebyshev 逼近.
    
    返回:
        (系数数组 c, beta2_min, beta2_max)
    """
    from potential_energy_surface import potential_energy
    
    def f(b_arr):
        vals = np.zeros(len(b_arr))
        for i in range(len(b_arr)):
            q = np.array([b_arr[i], 0.0, 0.0, 0.0, 0.0])
            vals[i] = potential_energy(q, mass_number, charge_number)
        return vals
    
    c = chebyshev_coefficients(f, n_terms, beta2_min, beta2_max)
    return c, beta2_min, beta2_max
