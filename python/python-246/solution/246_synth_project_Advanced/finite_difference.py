"""
finite_difference.py  —  高阶有限差分算子与紧致差分格式
====================================================

科学来源种子:
  - 367_fd2d_heat_steady / interior.m, boundary.m
    直接使用其 5 点离散模板与边界注入思路,但提升到 3D Poisson,
    并引入高阶模板。
  - 979_r8gb / r8gb_fa.m, r8gb_sl.m, r8gb_dif2.m
    直接使用其二差分带结构造 (r8gb_dif2) 与带状 LU 分解/回代
    (r8gb_fa/r8gb_sl) 的算法,移植到 Python 用于 1D 紧致隐式求解。
  - 666_legendre_shifted_polynomial
    用于高阶窗函数重建的 Legendre 基。

物理背景:
  在 Particle-Mesh (PM) 方法中,求解 Poisson 方程:
      ∇² Φ = 4π G ρ̄ a² δ
  在均匀网格上,使用 2p 阶中心差分:
      (δ²_h u)_i / h² = Σ_{k=1}^{p} c_k (u_{i+k} - 2u_i + u_{i-k}) / h²
  其中 c_k 为高阶差分系数,通过 Taylor 展开匹配 ∂²u/∂x² 至 O(h^{2p})。

  紧致 (compact) 差分格式 (Lele 1992):
      β f''_{i-1} + f''_i + β f''_{i+1}
          = a (f_{i+1} - 2f_i + f_{i-1})/h²
          + b (f_{i+2} - 2f_i + f_{i-2})/(2h)²
          + c (f_{i+3} - 2f_i + f_{i-3})/(3h)²
  对本原三对角系统,使用 Thomas 算法 (即 r8gb_sl 的 1D 版本) 求解。
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


# ---------------------------------------------------------------------------- #
#                       高阶中心差分系数 (2p 阶)
# ---------------------------------------------------------------------------- #
def fd_coefficients_2nd_derivative(p: int) -> NDArray:
    """
    返回 2p 阶中心差分 ∂²/∂x² 的系数 c_1, ..., c_p,使得:
        f''(x) ≈ (1/h²) [ -2 Σ_{k=1}^p c_k f(x)
                            + Σ_{k=1}^p c_k (f(x+kh) + f(x-kh)) ]
    通过解 Vandermonde 系统:
        Σ_{k=1}^p c_k k^{2m} = δ_{m,1},   m = 1, 2, ..., p
    例如:
        p=1 (2阶): c_1 = 1        → 标准三点模板 [1, -2, 1]
        p=2 (4阶): c_1 = 4/3, c_2 = -1/12
        p=3 (6阶): c_1 = 3/2, c_2 = -3/20, c_3 = 1/90
        p=4 (8阶): c_1 = 8/5, c_2 = -1/5, c_3 = 4/315, c_4 = -1/560

    Parameters
    ----------
    p : int
        半带宽 (p=1 → 2 阶; p=2 → 4 阶; p=3 → 6 阶; p=4 → 8 阶)

    Returns
    -------
    c : (p,) array
    """
    if p < 1 or p > 4:
        raise ValueError(f"p={p} 必须 ∈ [1, 4]")
    # 解 Vandermonde 系统: Σ_{k=1}^p c_k k^{2m} = δ_{m,1}  (m=1..p)
    A = np.zeros((p, p))
    rhs = np.zeros(p)
    rhs[0] = 1.0
    for m in range(1, p + 1):
        for k in range(1, p + 1):
            A[m - 1, k - 1] = k ** (2 * m)
    c = np.linalg.solve(A, rhs)
    return c


def fd_stencil_2nd(p: int) -> NDArray:
    """
    返回 2p 阶二阶导数完整模板 (长度 2p+1),中心在索引 p:
        stencil = [c_p, c_{p-1}, ..., c_1, -2 Σ c_k, c_1, ..., c_p]
    满足 Σ stencil = 0 (常数函数二阶导为零)。
    """
    c = fd_coefficients_2nd_derivative(p)
    s = np.zeros(2 * p + 1)
    s[p] = -2.0 * np.sum(c)
    for k in range(1, p + 1):
        s[p - k] = c[k - 1]
        s[p + k] = c[k - 1]
    return s


# ---------------------------------------------------------------------------- #
#                  紧致 (compact) 差分格式的系数
# ---------------------------------------------------------------------------- #
def compact_fd_coefficients(scheme: str = "c4") -> dict:
    """
    返回紧致差分格式系数 (Lele 1992, JCP 103, 16):
    形式:
        β f''_{i-1} + f''_i + β f''_{i+1}
          = a (f_{i+1}-2f_i+f_{i-1})/h²
          + b (f_{i+2}-2f_i+f_{i-2})/(4h²)
          + c (f_{i+3}-2f_i+f_{i-3})/(9h²)

    Parameters
    ----------
    scheme : str in {"c4", "c6", "c8"}
        c4: β=0, a=1, b=c=0 → 退化显式 2 阶
        c4: β=1/10, a=3/2·(1+1/5·...), 4 阶紧致
        c6: 6 阶紧致
        c8: 8 阶紧致

    Returns
    -------
    dict with keys {alpha, beta, a, b, c} 其中 alpha 为 f' 格式的一阶参数
    """
    if scheme == "c4":
        # 4 阶紧致三对角: β f''_{i-1} + f''_i + β f''_{i+1}
        #                 = (3/2)(f_{i+1}-2f_i+f_{i-1})/h²
        #                 with β = 1/10  → O(h^4)
        return {"beta": 1.0 / 10.0, "a": 6.0 / 5.0, "b": 0.0, "c": 0.0,
                "alpha": 1.0 / 4.0}
    if scheme == "c6":
        # 6 阶紧致: β = 2/11, a = 12/11 · (1 + 1/12), b = 3/11 · 1/16
        return {"beta": 2.0 / 11.0,
                "a": 12.0 / 11.0 * (1.0 + 1.0 / 12.0),
                "b": 3.0 / 11.0 / 16.0,
                "c": 0.0,
                "alpha": 1.0 / 3.0}
    if scheme == "c8":
        return {"beta": 1.0 / 7.0,
                "a": 8.0 / 7.0 * (1.0 + 1.0 / 16.0 + 1.0 / 128.0),
                "b": 4.0 / 7.0 * (1.0 / 16.0 + 1.0 / 32.0),
                "c": 1.0 / 7.0 / 144.0,
                "alpha": 1.0 / 2.0}
    raise ValueError(f"未知紧致格式 scheme={scheme}")


# ---------------------------------------------------------------------------- #
#               带状矩阵 Thomas 算法 (源自 r8gb_sl)
# ---------------------------------------------------------------------------- #
def thomas_solve(a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> NDArray:
    """
    三对角系统 Ax = d 的 Thomas 算法 (即 r8gb_sl 在 ml=mu=1 的特化)。

    A = | b_0  c_0                       |
        | a_1  b_1  c_1                  |
        |      a_2  b_2  c_2             |
        |            ...   ...   ...     |
    """
    n = b.size
    if a.size != n or c.size != n or d.size != n:
        raise ValueError("三对角向量尺寸不一致")
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    if abs(b[0]) < 1e-30:
        raise ValueError("Thomas 算法首主元为零,系统奇异")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-30:
            raise ValueError(f"Thomas 算法在第 {i} 步遇到零主元")
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ---------------------------------------------------------------------------- #
#               带状 LU 分解 (源自 r8gb_fa / r8gb_trf)
# ---------------------------------------------------------------------------- #
def r8gb_fa_python(n: int, ml: int, mu: int, a: NDArray) -> Tuple[NDArray, NDArray, int]:
    """
    复现 r8gb_fa.m: LINPACK 风格带状矩阵 PLU 分解。
    输入:
        n  : 矩阵阶数
        ml : 下次带宽
        mu : 上次带宽
        a  : (2*ml+mu+1, n) 带存储矩阵
    返回:
        alu   : (2*ml+mu+1, n) LU 因子
        pivot : (n,) 主元索引
        info  : 0 成功, >0 第 info 步奇异
    """
    m = ml + mu + 1
    alu = a.copy().astype(float)
    pivot = np.zeros(n, dtype=int)
    info = 0
    # 清零初始 fill-in 列
    j0 = mu + 2
    j1 = min(n, m) - 1
    for jz in range(j0 - 1, j1):
        i0 = m - jz
        if i0 < 1:
            i0 = 1
        alu[i0 - 1:ml, jz] = 0.0
    jz = j1 - 1
    ju = 0
    for k in range(n - 1):
        jz += 1
        if jz < n:
            alu[0:ml, jz] = 0.0
        lm = min(ml, n - k - 1)
        # 找主元
        l_idx = m - 1
        for j in range(m, m + lm):
            if abs(alu[l_idx, k]) < abs(alu[j, k]):
                l_idx = j
        pivot[k] = l_idx - m + 1 + k
        if abs(alu[l_idx, k]) < 1e-300:
            info = k + 1
            return alu, pivot, info
        # 交换
        if l_idx != m - 1:
            alu[[l_idx, m - 1], k] = alu[[m - 1, l_idx], k]
        # 计算乘子
        if lm > 0:
            alu[m:m + lm, k] /= -alu[m - 1, k]
        ju_k = max(ju, mu + pivot[k])
        ju_k = min(ju_k, n - 1)
        mm = m - 1
        for j in range(k + 1, ju_k + 1):
            l_off = l_idx - (m - 1)
            jj = j - (k + 1)
            src_row = mm - l_off
            if 0 <= src_row < 2 * ml + mu + 1:
                alu[src_row, j], alu[mm, j] = alu[mm, j], alu[src_row, j]
            for ii in range(lm):
                alu[mm + 1 + ii, j] += alu[mm, j] * alu[m + ii, k]
            mm -= 1
        ju = ju_k
    pivot[n - 1] = n - 1
    if abs(alu[m - 1, n - 1]) < 1e-300:
        info = n
    return alu, pivot, info


def r8gb_sl_python(n: int, ml: int, mu: int, alu: NDArray,
                   pivot: NDArray, b: NDArray, job: int = 0) -> NDArray:
    """
    复现 r8gb_sl.m: 求解已 PLU 分解的带状系统。
    job = 0 → A x = b ;  job ≠ 0 → A^T x = b
    """
    m = mu + ml + 1
    x = b.copy().astype(float)
    if job == 0:
        # L y = b
        if ml >= 1:
            for k in range(n - 1):
                lm = min(ml, n - k - 1)
                l = pivot[k]
                if l != k:
                    x[l], x[k] = x[k], x[l]
                for i in range(lm):
                    x[k + 1 + i] += x[k] * alu[m + i, k]
        # U x = y
        for k in range(n - 1, -1, -1):
            if abs(alu[m - 1, k]) < 1e-300:
                raise ValueError("回代遇到零主元")
            x[k] /= alu[m - 1, k]
            lm = min(k, m - 1)
            la = m - 1 - lm
            lb = k - lm
            for i in range(lm):
                x[lb + i] -= x[k] * alu[la + i, k]
    else:
        # U^T y = b
        for k in range(n):
            lm = min(k, m - 1)
            la = m - 1 - lm
            lb = k - lm
            for i in range(lm):
                x[k] -= alu[la + i, k] * x[lb + i]
            if abs(alu[m - 1, k]) < 1e-300:
                raise ValueError("转置回代遇到零主元")
            x[k] /= alu[m - 1, k]
        # L^T x = y
        if ml >= 1:
            for k in range(n - 2, -1, -1):
                lm = min(ml, n - k - 1)
                for i in range(lm):
                    x[k] += alu[m + i, k] * x[k + 1 + i]
                l = pivot[k]
                if l != k:
                    x[l], x[k] = x[k], x[l]
    return x


# ---------------------------------------------------------------------------- #
#            1D 紧致隐式二阶导数求解
# ---------------------------------------------------------------------------- #
def compact_second_derivative_1d(f: NDArray, h: float,
                                 scheme: str = "c4") -> NDArray:
    """
    使用紧致格式求解 f''(x) 在周期域上的近似:
        β f''_{i-1} + f''_i + β f''_{i+1} = RHS_i
    RHS_i = a (f_{i+1}-2f_i+f_{i-1})/h² + b (f_{i+2}-2f_i+f_{i-2})/(4h²) + ...
    使用周期 Thomas 算法 (Sherman-Morrison) 求解。
    """
    co = compact_fd_coefficients(scheme)
    beta = co["beta"]
    a = co["a"]
    b = co["b"]
    c = co["c"]
    n = f.size
    rhs = np.zeros(n)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        ip2 = (i + 2) % n
        im2 = (i - 2) % n
        ip3 = (i + 3) % n
        im3 = (i - 3) % n
        d1 = (f[ip1] - 2.0 * f[i] + f[im1]) / (h * h)
        d2 = (f[ip2] - 2.0 * f[i] + f[im2]) / (4.0 * h * h)
        d3 = (f[ip3] - 2.0 * f[i] + f[im3]) / (9.0 * h * h)
        rhs[i] = a * d1 + b * d2 + c * d3
    if abs(beta) < 1e-14:
        return rhs
    # 周期三对角: A = tridiag(β, 1, β) + corner corrections
    # Sherman-Morrison: A = B + u v^T
    gamma = -1.0  # 任意非零,使 B 非奇异
    b_diag = np.ones(n)
    b_lower = np.full(n, beta)
    b_upper = np.full(n, beta)
    b_diag[0] = b_diag[0] - gamma
    b_diag[n - 1] = b_diag[n - 1] - beta * beta / gamma
    u = np.zeros(n)
    v = np.zeros(n)
    u[0] = gamma
    u[n - 1] = beta
    v[0] = 1.0
    v[n - 1] = beta / gamma
    # 解 B z = rhs 与 B q = u
    z = _tridiag_solve(b_lower, b_diag, b_upper, rhs)
    q = _tridiag_solve(b_lower, b_diag, b_upper, u)
    vdotz = np.dot(v, z)
    vdotq = np.dot(v, q)
    if abs(1.0 + vdotq) < 1e-14:
        raise ValueError("Sherman-Morrison 分母接近零,系统奇异")
    x = z - q * (vdotz / (1.0 + vdotq))
    return x


def _tridiag_solve(a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> NDArray:
    return thomas_solve(a, b, c, d)


# ---------------------------------------------------------------------------- #
#                  3D 拉普拉斯算子 (高阶)
# ---------------------------------------------------------------------------- #
def laplacian_3d_periodic(u: NDArray, h: float, p: int = 2) -> NDArray:
    """
    在周期 3D 网格上计算 ∇² u 的 2p 阶中心差分近似:
        ∇² u = (∂²/∂x² + ∂²/∂y² + ∂²/∂z²) u
    使用分离式高阶模板:
        (∂²u/∂x²)_{ijk} = (1/h²) Σ_{m=-p}^p s_{p+m} u_{i+m,j,k}

    Parameters
    ----------
    u : (N, N, N) array
    h : 网格间距
    p : 半带宽,2p 阶精度 (p=1: 2阶; p=2: 4阶; p=3: 6阶; p=4: 8阶)
    """
    if u.ndim != 3:
        raise ValueError(f"u.ndim={u.ndim} 必须为 3")
    nx, ny, nz = u.shape
    if nx != ny or ny != nz:
        raise ValueError("仅支持立方网格")
    s = fd_stencil_2nd(p)
    lap = np.zeros_like(u)
    # x 方向
    for m in range(-p, p + 1):
        lap += s[m + p] * np.roll(u, -m, axis=0)
    # y 方向
    for m in range(-p, p + 1):
        lap += s[m + p] * np.roll(u, -m, axis=1)
    # z 方向
    for m in range(-p, p + 1):
        lap += s[m + p] * np.roll(u, -m, axis=2)
    return lap / (h * h)


def gradient_3d_periodic(u: NDArray, h: float, p: int = 2) -> NDArray:
    """
    3D 周期域上 ∇u 的 2p 阶中心差分:
        (∂u/∂x)_{i} = (1/h) Σ_{m=-p}^p g_m u_{i+m}
    其中 g_m 为奇对称一阶导数模板。
    返回 (3, N, N, N) 数组。
    """
    g = fd_coefficients_1st_derivative(p)
    grad = np.zeros((3,) + u.shape)
    for axis in range(3):
        for m in range(1, p + 1):
            grad[axis] += g[m - 1] * (np.roll(u, -m, axis=axis)
                                      - np.roll(u, m, axis=axis))
    return grad / h


def fd_coefficients_1st_derivative(p: int) -> NDArray:
    """
    2p 阶中心一阶导数模板系数 g_1, ..., g_p:
        (∂u/∂x)_i ≈ (1/h) Σ_{k=1}^p g_k (u_{i+k} - u_{i-k})
    通过解 Vandermonde 系统:
        Σ_{k=1}^p g_k k^{2m-1} = δ_{m,1}/2,   m=1..p
    例如:
        p=1 (2阶): g_1 = 1/2
        p=2 (4阶): g_1 = 2/3, g_2 = -1/12
        p=3 (6阶): g_1 = 3/4, g_2 = -3/20, g_3 = 1/60
        p=4 (8阶): g_1 = 4/5, g_2 = -1/5, g_3 = 4/105, g_4 = -1/280
    """
    if p < 1 or p > 4:
        raise ValueError(f"p={p} 必须 ∈ [1, 4]")
    A = np.zeros((p, p))
    rhs = np.zeros(p)
    rhs[0] = 0.5
    for m in range(1, p + 1):
        for k in range(1, p + 1):
            A[m - 1, k - 1] = k ** (2 * m - 1)
    g = np.linalg.solve(A, rhs)
    return g
