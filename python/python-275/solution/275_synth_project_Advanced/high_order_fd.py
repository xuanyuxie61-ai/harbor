"""
high_order_fd.py
================
高阶有限差分算子构造与非厄米 PDE 空间离散化.
融合种子项目:
  - 357_fd1d_burgers_leap: leapfrog 时间推进 + 中心差分
  - 003_allen_cahn_pde: 区间上的 Laplacian 离散

物理背景:
  非厄米扩散-反应方程 (连续):
    ∂ψ/∂t = ν ∂²ψ/∂x² + iγ(x)ψ - V(ψ)
  其中 γ(x) 为非厄米在位势 (增益/损耗分布), V(ψ) 为非线性项.
  空间离散采用 2p 阶中心差分:
    d²ψ/dx²|_j ≈ (1/dx²) sum_{m=-p}^{p} c_m ψ_{j+m}
  其中 c_m 由 Taylor 展开匹配至 O(dx^{2p}).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from fractions import Fraction


# ---------------------------------------------------------------------------
# 高阶差分系数 (源自 357_fd1d_burgers_leap 的差分思想 + 003 的 Laplacian)
# ---------------------------------------------------------------------------
def fd_coefficients_second_derivative(p: int = 2) -> np.ndarray:
    """构造 2p 阶精度的二阶导数中心差分系数.

    通过求解 Vandermonde 系统:
      sum_{m=1}^{p} c_m * m^{2(j+1)} = delta_{j,0},  j = 0, 1, ..., p-1
    给出对称系数 c_{-m} = c_m, c_0 = -2*sum_{m>0} c_m.

    数学推导:
      Taylor 展开 ψ(x±mh) 相加:
        ∑_{m=1}^p c_m [ψ(x+mh) + ψ(x-mh)] + c_0 ψ(x)
        = (2∑c_m + c_0) ψ + h²(∑c_m m²) ψ'' + (h⁴/12)(∑c_m m⁴) ψ⁽⁴⁾ + ...
      要求 (2∑c_m+c_0)=0, ∑c_m m²=1, ∑c_m m⁴=0, ..., ∑c_m m^{2p}=0.
      得到 p 个方程 (j=0..p-1): ∑c_m m^{2(j+1)} = δ_{j,0}.

    Parameters
    ----------
    p : 半带宽, 精度 O(dx^{2p}).

    Returns
    -------
    c : (2p+1,) 系数数组, c[p] 为中心点.
    """
    if p < 1 or p > 8:
        raise ValueError("p must be in [1, 8]")
    # 构造 Vandermonde 系统: A[j, m] = m^{2(j+1)}, j=0..p-1, m=1..p
    A = np.zeros((p, p))
    rhs = np.zeros(p)
    rhs[0] = 1.0  # 匹配二阶导数: ∑c_m m^2 = 1
    for j in range(p):
        for m_idx in range(p):
            m = m_idx + 1
            A[j, m_idx] = m ** (2 * (j + 1))
    c_pos = np.linalg.solve(A, rhs)
    c = np.zeros(2 * p + 1)
    for m_idx in range(p):
        m = m_idx + 1
        c[p - m] = c_pos[m_idx]
        c[p + m] = c_pos[m_idx]
    c[p] = -2.0 * np.sum(c_pos)
    return c


def apply_laplacian_fd(u: np.ndarray, dx: float,
                       order: int = 4,
                       bc: str = "periodic") -> np.ndarray:
    """对一维数组 u 施加高阶 Laplacian.

    边界处理:
      - periodic: 环绕
      - dirichlet: u[0]=u[-1]=0, 仅内部点计算
      - neumann: du/dx = 0 边界, 通过镜像延拓

    Parameters
    ----------
    u : (N,) 实或复数组.
    dx : 网格间距.
    order : 差分精度 (2 或 4).
    bc : 边界条件.

    Returns
    -------
    uxx : (N,) Laplacian 近似.
    """
    N = len(u)
    if N < 5 and order == 4:
        order = 2
    p = order // 2
    c = fd_coefficients_second_derivative(p)
    uxx = np.zeros_like(u, dtype=complex if np.iscomplexobj(u) else float)
    if bc == "periodic":
        u_ext = np.concatenate([u[-p:], u, u[:p]])
    elif bc == "dirichlet":
        u_ext = np.zeros(N + 2 * p, dtype=u.dtype)
        u_ext[p:p + N] = u
    elif bc == "neumann":
        u_ext = np.concatenate([u[:p][::-1], u, u[-p:][::-1]])
    else:
        raise ValueError(f"unknown bc: {bc}")
    for j in range(N):
        val = 0.0
        for m in range(-p, p + 1):
            val += c[m + p] * u_ext[j + p + m]
        uxx[j] = val / (dx * dx)
    if bc == "dirichlet":
        uxx[0] = 0.0
        uxx[-1] = 0.0
    return uxx


def fd_first_derivative(u: np.ndarray, dx: float,
                        order: int = 4,
                        bc: str = "periodic") -> np.ndarray:
    """中心差分一阶导数 (源自 Burgers leapfrog 的中心差分).

    2p 阶精度: du/dx|_j ≈ (1/dx) ∑_{m=-p}^{p} d_m u_{j+m}
    其中 d_m 为奇函数: d_{-m} = -d_m, d_0 = 0.

    数学: Taylor 展开给出约束:
      ∑_{m=1}^p d_m * m = 1/2 (匹配一阶导数)
      ∑_{m=1}^p d_m * m³ = 0 (消除三阶项)
      ∑_{m=1}^p d_m * m^{2p-1} = 0
    """
    N = len(u)
    p = max(1, order // 2)
    if N < 2 * p + 1:
        p = 1
    # 构造 Vandermonde 系统: A[j, m] = m^{2j+1}, rhs[0] = 1/2
    A = np.zeros((p, p))
    rhs = np.zeros(p)
    rhs[0] = 0.5
    for j in range(p):
        for m_idx in range(p):
            m = m_idx + 1
            A[j, m_idx] = m ** (2 * j + 1)
    d_pos = np.linalg.solve(A, rhs)
    d = np.zeros(2 * p + 1)
    for m_idx in range(p):
        m = m_idx + 1
        d[p + m] = d_pos[m_idx]
        d[p - m] = -d_pos[m_idx]
    if bc == "periodic":
        u_ext = np.concatenate([u[-p:], u, u[:p]])
    elif bc == "dirichlet":
        u_ext = np.zeros(N + 2 * p, dtype=u.dtype)
        u_ext[p:p + N] = u
    else:
        u_ext = np.concatenate([u[:p][::-1], u, u[-p:][::-1]])
    ux = np.zeros_like(u, dtype=complex if np.iscomplexobj(u) else float)
    for j in range(N):
        val = 0.0
        for m in range(-p, p + 1):
            val += d[m + p] * u_ext[j + p + m]
        ux[j] = val / dx
    return ux


# ---------------------------------------------------------------------------
# 非厄米 PDE 空间离散: 构建大矩阵 H_fd
# ---------------------------------------------------------------------------
def build_nonhermitian_fd_matrix(N: int, dx: float, nu: float,
                                 gamma_profile: np.ndarray,
                                 order: int = 4,
                                 bc: str = "periodic") -> np.ndarray:
    """构造非厄米扩散算子的矩阵形式.

    H_fd = ν * D2 + diag(i*γ(x))
    其中 D2 为高阶二阶差分矩阵.

    数学表达:
      H_fd ψ = ν ∑_m c_m/dx² ψ_{j+m} + i γ_j ψ_j
    非厄米性体现在 γ(x) 为实数但乘以 i.
    """
    if len(gamma_profile) != N:
        raise ValueError("gamma_profile length must equal N")
    H = np.zeros((N, N), dtype=complex)
    p = order // 2
    c = fd_coefficients_second_derivative(p)
    for j in range(N):
        for m in range(-p, p + 1):
            if bc == "periodic":
                j_m = (j + m) % N
            elif bc == "dirichlet":
                j_m = j + m
                if j_m < 0 or j_m >= N:
                    continue
            else:
                j_m = j + m
                if j_m < 0:
                    j_m = -j_m
                elif j_m >= N:
                    j_m = 2 * (N - 1) - j_m
            if 0 <= j_m < N:
                H[j, j_m] += nu * c[m + p] / (dx * dx)
        H[j, j] += 1j * gamma_profile[j]
    if bc == "dirichlet":
        H[0, :] = 0.0
        H[-1, :] = 0.0
        H[0, 0] = 1.0
        H[-1, -1] = 1.0
    return H


# ---------------------------------------------------------------------------
# Leapfrog 时间推进 (源自 357_fd1d_burgers_leap)
# ---------------------------------------------------------------------------
def leapfrog_step(u_old: np.ndarray, u_cur: np.ndarray,
                  rhs_func, dt: float) -> np.ndarray:
    """Leapfrog 时间步进: u^{n+1} = u^{n-1} + 2*dt*RHS(u^n).

    用于非厄米薛定谔方程的时间演化.
    注意: leapfrog 对纯虚数本征值不稳定, 需结合耗散或隐式方法.
    """
    return u_old + 2.0 * dt * rhs_func(u_cur)


def cauchy_theta_step(u_n: np.ndarray, rhs_func,
                      dt: float, theta: float = 0.5,
                      it_max: int = 5) -> np.ndarray:
    """Cauchy theta 方法 (源自 138_cauchy_method).

    中间点: u_m = u_n + θ*dt*RHS(u_m)  (不动点迭代)
    下一步: u_{n+1} = (1/θ)*u_m + (1 - 1/θ)*u_n

    对非厄米系统, θ=1 为后向 Euler (稳定), θ=0.5 为 Crank-Nicolson 型.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError("theta must be in (0, 1]")
    u_m = u_n.copy()
    for _ in range(it_max):
        u_m = u_n + theta * dt * rhs_func(u_m)
    u_next = (1.0 / theta) * u_m + (1.0 - 1.0 / theta) * u_n
    return u_next


# ---------------------------------------------------------------------------
# Von Neumann 稳定性分析
# ---------------------------------------------------------------------------
def von_neumann_growth_factor(kx: float, dx: float, dt: float,
                              nu: float, gamma: float,
                              scheme: str = "leapfrog",
                              order: int = 2) -> complex:
    """计算给定波数 kx 的增长因子 g(kx).

    对方案 ∂u/∂t = ν ∂²u/∂x² + iγu, Fourier 模式 exp(ikx) 的半离散系统:
      d û / dt = (-ν kx² + iγ) û
    离散后, 差分算子的符号本征值为:
      λ_fd(kx) = (1/dx²) ∑_m c_m exp(i m kx dx)

    Leapfrog: g = -i*dt*λ_fd ± sqrt(1 - (dt*λ_fd)²)
    |g| > 1 表示不稳定.
    """
    p = order // 2
    c = fd_coefficients_second_derivative(p)
    lambda_fd = 0.0 + 0.0j
    for m in range(-p, p + 1):
        lambda_fd += c[m + p] * np.exp(1j * m * kx * dx)
    lambda_fd *= nu / (dx * dx)
    lambda_fd += 1j * gamma
    z = dt * lambda_fd
    if scheme == "leapfrog":
        disc = 1.0 - z * z
        sqrt_disc = np.sqrt(disc + 0.0j)
        g1 = -1j * z + sqrt_disc
        g2 = -1j * z - sqrt_disc
        return max(abs(g1), abs(g2))
    elif scheme == "cauchy":
        return abs(1.0 + z)
    else:
        raise ValueError(f"unknown scheme: {scheme}")


def stability_domain_scan(nu: float, gamma: float, dx: float,
                          r_range: np.ndarray,
                          order: int = 2,
                          scheme: str = "leapfrog") -> np.ndarray:
    """扫描 CFL 数 r = ν*dt/dx², 计算最大增长因子.

    Returns
    -------
    g_max : (len(r_range),) 每个 r 下 max_kx |g(kx)|.
    """
    g_max = np.zeros(len(r_range))
    n_k = 128
    kx_vals = np.linspace(0, np.pi / dx, n_k)
    for idx, r in enumerate(r_range):
        dt = r * dx * dx / max(abs(nu), 1e-12)
        g_local = 0.0
        for kx in kx_vals:
            g = von_neumann_growth_factor(kx, dx, dt, nu, gamma,
                                          scheme=scheme, order=order)
            if g > g_local:
                g_local = g
        g_max[idx] = g_local
    return g_max
