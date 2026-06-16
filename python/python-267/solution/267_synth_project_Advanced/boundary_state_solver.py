"""
boundary_state_solver.py — 拓扑绝缘体边界态求解器
===================================================

本模块实现拓扑绝缘体边界态 (边缘态/表面态) 的数值求解, 核心方法包括:

1. **表面格林函数迭代法** (Iterative Surface Green's Function):
   通过递归 Renormalization Group (RG) 方法计算半无限系统的
   表面格林函数 G_s(E), 其谱函数揭示边界态的存在和色散。

2. **转移矩阵法** (Transfer Matrix Method):
   对 ribbon 几何 (有限宽度, 周期方向), 直接对角化
   得到边界态本征值和本征矢。

3. **边界态定位长度提取**:
   从表面格林函数的衰减行为提取边界态的局域化长度 ξ。

物理公式
--------
**表面格林函数 (迭代法):**
    G_s(E) = [(E+iη)I - H₀₀ - H₀₁ · g_s(E) · H₁₀]⁻¹

    其中迭代:
        t_s = H₁₀ · [(E+iη)I - H₀₀]⁻¹ · H₀₁
        t̄_s = H₀₁ · [(E+iη)I - H₀₀]⁻¹ · H₁₀

    Sancho-Rubio 迭代:
        T_n = T_{n-1} · (I - t_{n-1}·t̄_{n-1})⁻¹ · t_{n-1}
        t̄_n = t̄_{n-1} + t̄_{n-1} · (I - t_{n-1}·t̄_{n-1})⁻¹ · t_{n-1}·t̄_{n-1}
        (类似的 T̄_n 迭代)

    收敛后: G_s = [(E+iη)I - H₀₀ - H₀₁ · T_total]⁻¹

**谱函数:**
    A(k, E) = -(1/π) Im[Tr G_s(k, E)]

**边界态定位长度:**
    ξ(E) = -d / ln|λ_min|
    其中 λ_min 是转移矩阵的最小本征值的模, d 是层间距。

**边缘态电导 (Landauer-Büttiker):**
    G = (e²/h) × N_channels
    对量子自旋霍尔态: G = 2e²/h (两个自旋通道)

来源映射
--------
- 131_c8lib: 复数矩阵求逆 (高斯消元)
- 1209_brady-flinchum: 表面曲率的几何势修正
- 1099_amsontag: 多通道输运方程并行求解
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from typing import Tuple, Dict, List, Optional
from bhz_hamiltonian import BHZParameters, build_bhz_hamiltonian


def surface_green_function_iterative(H00: np.ndarray,
                                     H01: np.ndarray,
                                     E: complex,
                                     n_iter: int = 100,
                                     tol: float = 1e-10
                                     ) -> Tuple[np.ndarray, int, float]:
    """
    Sancho-Rubio 迭代法计算半无限系统的表面格林函数。

    将半无限系统沿 y 方向分层, 每层内 Hamiltonian 为 H00,
    相邻层间耦合为 H01 (前向) 和 H10 = H01† (后向)。

    迭代算法 (Sancho, Sancho, Rubio, J. Phys. F 15, 851 (1985)):

    初始化:
        ε₀ = H₀₀
        t₀ = H₀₁ · [(E+iη)I - H₀₀]⁻¹ · H₁₀
        t̄₀ = H₁₀ · [(E+iη)I - H₀₀]⁻¹ · H₀₁
        T₀ = H₀₁
        T̄₀ = H₁₀

    第 n 步迭代:
        G_n = [(E+iη)I - ε_n]⁻¹
        Δt_n = t_n · G_n · t̄_n
        Δt̄_n = t̄_n · G_n · t_n
        ε_{n+1} = ε_n + t_n · G_n · T̄_n + T_n · G_n · t̄_n  (简化)
        t_{n+1} = t_n · G_n · t_n
        t̄_{n+1} = t̄_n · G_n · t̄_n
        T_{n+1} = T_n · G_n · t_n
        T̄_{n+1} = T̄_n · G_n · t̄_n

    收敛条件: ||t_n|| < tol 且 ||t̄_n|| < tol

    表面格林函数: G_s = [(E+iη)I - ε_∞]⁻¹

    Parameters
    ----------
    H00 : ndarray, shape (m, m)
        层内哈密顿量
    H01 : ndarray, shape (m, m)
        层间耦合 (前向)
    E : complex
        能量 (含虚部 iη)
    n_iter : int
        最大迭代次数
    tol : float
        收敛阈值

    Returns
    -------
    G_s : ndarray, shape (m, m)
        表面格林函数
    n_converged : int
        收敛时的迭代次数 (-1 表示未收敛)
    residual : float
        最终残差 ||t|| + ||t̄||
    """
    m = H00.shape[0]
    I_m = np.eye(m, dtype=np.complex128)
    H10 = H01.conj().T

    # 初始化
    zI_E = E * I_m
    g0 = np.linalg.inv(zI_E - H00)  # [(E+iη)I - H00]⁻¹

    eps = H00.copy()
    t = H01 @ g0 @ H10
    t_bar = H10 @ g0 @ H01
    T = H01.copy()
    T_bar = H10.copy()

    n_converged = -1
    residual = np.inf

    for n in range(n_iter):
        # 检查收敛
        norm_t = np.linalg.norm(t, 'fro')
        norm_tbar = np.linalg.norm(t_bar, 'fro')
        residual = norm_t + norm_tbar

        if residual < tol:
            n_converged = n
            break

        # 迭代步
        g_n = np.linalg.inv(zI_E - eps)

        # 更新转移量
        t_new = t @ g_n @ t
        t_bar_new = t_bar @ g_n @ t_bar
        T_new = T @ g_n @ t
        T_bar_new = T_bar @ g_n @ t_bar

        # 更新有效层内哈密顿量
        eps = eps + t @ g_n @ T_bar + T @ g_n @ t_bar

        t = t_new
        t_bar = t_bar_new
        T = T_new
        T_bar = T_bar_new

        # 数值安全检查
        if not np.all(np.isfinite(eps)):
            # 发散, 返回当前最佳估计
            break

    # 计算表面格林函数
    G_s = np.linalg.inv(zI_E - eps)

    return G_s, n_converged, residual


def spectral_function(G_s: np.ndarray) -> float:
    """
    从表面格林函数计算谱函数 A(E) = -(1/π) Im[Tr G_s(E)]。

    谱函数的峰对应边界态或共振态的能量位置。

    Parameters
    ----------
    G_s : ndarray
        表面格林函数

    Returns
    -------
    A : float
        谱函数值
    """
    return -np.imag(np.trace(G_s)) / np.pi


def localization_length(G_s: np.ndarray, d: float) -> float:
    """
    从表面格林函数估计边界态的局域化长度。

    ξ = -d / ln(|G_s[0,0]| / |G_s[0,N-1]|)

    简化估计: 使用格林函数对角元的衰减率。
    更精确的方法需要层分辨的格林函数。

    Parameters
    ----------
    G_s : ndarray
        表面格林函数
    d : float
        层间距 (nm)

    Returns
    -------
    xi : float
        局域化长度 (nm), np.inf 表示扩展态
    """
    m = G_s.shape[0]
    if m < 2:
        return np.inf

    # 使用对角元和角元的比值估计衰减
    diag_mean = np.mean(np.abs(np.diag(G_s)))
    if diag_mean < 1e-30:
        return np.inf

    # 非对角元的平均衰减
    off_diag = np.abs(G_s[0, -1]) if m > 1 else diag_mean
    ratio = off_diag / diag_mean

    if ratio >= 1.0 or ratio < 1e-30:
        return np.inf if ratio >= 1.0 else d

    xi = -d * (m - 1) / np.log(ratio)
    return max(xi, d)  # 不小于层间距


def ribbon_eigenvalues(Ny: int, hy: float, params: BHZParameters,
                       kx: float, p: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 ribbon 几何中给定 kx 的本征值谱。

    Ribbon: x 方向周期 (kx 好量子数), y 方向有限 (Ny 格点, 开边界)。

    对每个 kx, 哈密顿量为 4Ny × 4Ny 矩阵。
    边界态表现为连接导带和价带的色散曲线。

    Parameters
    ----------
    Ny : int
        y 方向格点数
    hy : float
        y 方向格距 (nm)
    params : BHZParameters
        BHZ 参数
    kx : float
        x 方向波矢 (nm⁻¹)
    p : int
        FD 半带宽

    Returns
    -------
    eigenvalues : ndarray, shape (4Ny,)
        本征值 (升序)
    eigenvectors : ndarray, shape (4Ny, 4Ny)
        本征矢 (列向量)
    """
    from fd_stencils import (first_derivative_coefficients,
                             second_derivative_coefficients)

    c1 = first_derivative_coefficients(p)
    d2, d2_diag = second_derivative_coefficients(p)

    # 构建 1D y 方向的 FD 算子 (开边界)
    dim_y = Ny
    Dy = np.zeros((dim_y, dim_y), dtype=np.complex128)
    Dyy = np.zeros((dim_y, dim_y), dtype=np.complex128)

    for i in range(dim_y):
        Dyy[i, i] = d2_diag / (hy * hy)
        for j_idx in range(p):
            j = j_idx + 1
            ip = i + j
            im = i - j
            if 0 <= ip < dim_y:
                Dy[i, ip] += c1[j_idx] / hy
                Dyy[i, ip] += d2[j_idx] / (hy * hy)
            if 0 <= im < dim_y:
                Dy[i, im] -= c1[j_idx] / hy
                Dyy[i, im] += d2[j_idx] / (hy * hy)

    # 对于给定 kx, 构建 4Ny × 4Ny Hamiltonian
    # kx → kx (精确), ky → -iDy, k² → kx² + (-Dyy)
    k2_val = kx * kx  # 标量部分

    # 各算子在 y 空间的作用
    # ε(k) = C - D(kx² - Dyy) = C - D·kx² + D·Dyy
    # M(k) = M0 - B(kx² - Dyy) = M0 - B·kx² + B·Dyy
    # A·kx → A·kx (标量)
    # A·ky → -iA·Dy

    I_Ny = np.eye(dim_y, dtype=np.complex128)
    Z_Ny = np.zeros((dim_y, dim_y), dtype=np.complex128)

    eps_op = (params.C - params.D * k2_val) * I_Ny + params.D * Dyy
    M_op = (params.M0 - params.B * k2_val) * I_Ny + params.B * Dyy

    # h(kx, Dy) 的块结构
    # h11 = ε + M
    # h12 = A(kx - i(-iDy)) = A(kx - Dy)  (注意: ky→-iDy, -iky→Dy)
    # h21 = A(kx + i(-iDy)) = A(kx + Dy)

    # 等一下, 让我重新推导:
    # A(kx - iky) → A(kx - i(-iDy)) = A(kx - Dy)
    # A(kx + iky) → A(kx + i(-iDy)) = A(kx + Dy)

    h12_op = params.A * (kx * I_Ny - Dy)
    h21_op = params.A * (kx * I_Ny + Dy)

    h11 = eps_op + M_op
    h22 = eps_op - M_op

    # 下块 h*(-k): kx→-kx, Dy→-Dy (因为 ky→-iDy, 取复共轭后 Dy→Dy*)
    # 但 Dy 是实矩阵 (开边界 FD), 所以 Dy* = Dy
    # h*(-k) 的推导:
    # h(-kx, -ky): kx→-kx, Dy→-Dy
    # h(-kx, -Dy):
    #   h11' = ε(-kx,-ky) + M(-kx,-ky) = (C-D(kx²+Dyy)) + (M0-B(kx²+Dyy))
    #        = 同 h11 (因为只依赖 k² 和 Dyy)
    #   h22' = ε - M = 同 h22
    #   h12' = A(-kx - (-Dy)) = A(-kx + Dy) = -A(kx - Dy) = -h12
    #   h21' = A(-kx + (-Dy)) = A(-kx - Dy) = -A(kx + Dy) = -h21
    # 取共轭: h12'* = -h12* = -h21 (因为 h12, h21 是实的)
    #          h21'* = -h21* = -h12

    h33 = eps_op + M_op
    h44 = eps_op - M_op
    h34 = -h21_op  # = -A(kx + Dy)
    h43 = -h12_op  # = -A(kx - Dy)

    # 组装 4Ny × 4Ny
    dim_total = 4 * dim_y
    H_ribbon = np.zeros((dim_total, dim_total), dtype=np.complex128)

    H_ribbon[0:dim_y, 0:dim_y] = h11
    H_ribbon[0:dim_y, dim_y:2*dim_y] = h12_op
    H_ribbon[dim_y:2*dim_y, 0:dim_y] = h21_op
    H_ribbon[dim_y:2*dim_y, dim_y:2*dim_y] = h22
    H_ribbon[2*dim_y:3*dim_y, 2*dim_y:3*dim_y] = h33
    H_ribbon[2*dim_y:3*dim_y, 3*dim_y:4*dim_y] = h34
    H_ribbon[3*dim_y:4*dim_y, 2*dim_y:3*dim_y] = h43
    H_ribbon[3*dim_y:4*dim_y, 3*dim_y:4*dim_y] = h44

    # 确保 Hermiticity
    H_ribbon = 0.5 * (H_ribbon + H_ribbon.conj().T)

    # 对角化
    eigenvalues, eigenvectors = np.linalg.eigh(H_ribbon)

    return eigenvalues, eigenvectors


def edge_state_dispersion(Ny: int, hy: float, params: BHZParameters,
                          kx_array: np.ndarray, p: int = 2
                          ) -> Dict:
    """
    计算边界态的完整色散关系 E(kx)。

    扫描 kx 从 -π/a 到 π/a, 对每个 kx 求解 ribbon 本征值问题。

    Parameters
    ----------
    Ny : int
        y 方向格点数
    hy : float
        格距 (nm)
    params : BHZParameters
    kx_array : ndarray
        kx 值数组
    p : int
        FD 半带宽

    Returns
    -------
    result : dict
        'kx': kx 数组
        'energies': shape (Nkx, 4Ny) 的本征值
        'bulk_gap': 体带隙估计
        'edge_states_in_gap': 带隙内的态的数量
    """
    Nkx = len(kx_array)
    dim = 4 * Ny
    energies = np.zeros((Nkx, dim))

    for ik, kx in enumerate(kx_array):
        evals, _ = ribbon_eigenvalues(Ny, hy, params, kx, p)
        energies[ik] = evals

    # 估计体带隙: 在 kx=0 处, 体带边在 E = C ± |M0| 附近
    # (有限尺寸会引入修正)
    bulk_gap_estimate = params.bulk_gap

    # 统计带隙内的态
    E_fermi = params.C  # 假设 Fermi 能级在带中心
    gap_half = bulk_gap_estimate / 2.0
    n_gap_states = 0
    for ik in range(Nkx):
        in_gap = np.sum(
            (energies[ik] > E_fermi - gap_half) &
            (energies[ik] < E_fermi + gap_half)
        )
        n_gap_states = max(n_gap_states, in_gap)

    return {
        'kx': kx_array,
        'energies': energies,
        'bulk_gap': bulk_gap_estimate,
        'edge_states_in_gap': n_gap_states,
        'ny': Ny,
        'hy': hy,
        'fd_order': 2 * p
    }


def compute_edge_conductance(energies: np.ndarray, kx_array: np.ndarray,
                             E_fermi: float, temperature: float = 0.001
                             ) -> Dict:
    """
    计算边缘态电导 (Landauer-Büttiker 公式)。

    G = (e²/h) Σ_n ∫ dkx (-∂f/∂E)|_{E=E_n(kx)} × v_n(kx)

    在零温下简化为计数 Fermi 能级处穿越的边界态通道数:
    G = (e²/h) × N_crossing

    其中 N_crossing 是 E_n(kx) = E_Fermi 的穿越次数 (考虑方向)。

    有限温度下使用 Fermi-Dirac 分布:
    f(E) = 1 / (1 + exp((E-μ)/(kB T)))

    Parameters
    ----------
    energies : ndarray, shape (Nkx, Nb)
        本征值
    kx_array : ndarray
    E_fermi : float
        Fermi 能级 (eV)
    temperature : float
        温度 (eV, kB=1 单位)

    Returns
    -------
    result : dict
    """
    Nkx, Nb = energies.shape

    # 计算群速度 v = dE/dkx
    dkx = kx_array[1] - kx_array[0] if Nkx > 1 else 1.0
    velocities = np.gradient(energies, dkx, axis=0)

    # Fermi-Dirac 分布
    def fermi(E, mu, T):
        x = (E - mu) / max(T, 1e-15)
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(x))

    # 计算传输函数 T(E_F) = Σ_n Σ_{kx: E_n(kx)=E_F} 1
    # 有限温度: 使用 -∂f/∂E 权重
    n_right = 0  # 向右传播的通道
    n_left = 0   # 向左传播的通道

    for ib in range(Nb):
        for ik in range(Nkx):
            E = energies[ik, ib]
            v = velocities[ik, ib]
            # -∂f/∂E 在 E = E_F 处的值
            dfdE_weight = np.exp((E - E_fermi) / max(temperature, 1e-15)) / \
                (max(temperature, 1e-15) *
                 (1.0 + np.exp((E - E_fermi) / max(temperature, 1e-15)))**2)

            if abs(E - E_fermi) < 5 * temperature:
                if v > 0:
                    n_right += dfdE_weight * dkx / (2 * np.pi)
                else:
                    n_left += abs(dfdE_weight) * dkx / (2 * np.pi)

    # Landauer 电导: G = (e²/h) × (N_right - N_left) 的净贡献
    # 对 QSH 态, 预期 N_right = N_left = 1 (每自旋一个通道)
    # 总电导 G = 2e²/h (考虑自旋)

    # 基本单位 e²/h ≈ 38.74 μS
    e2_over_h = 38.74e-6  # Siemens

    conductance_quanta = abs(n_right - n_left)

    return {
        'G_over_G0': conductance_quanta,
        'G_Siemens': conductance_quanta * e2_over_h,
        'n_right_channels': n_right,
        'n_left_channels': n_left,
        'quantized': abs(conductance_quanta - 2.0) < 0.5  # 预期 ≈ 2e²/h
    }


def surface_geometry_potential(curvature_K: float, params: BHZParameters,
                               h_grid: float) -> float:
    """
    计算表面曲率对边界态的几何势修正。

    对于弯曲的 TI 表面, 有效哈密顿量中出现几何势:
        V_geo = -(ℏ²/2m*) × (κ₁² + κ₂²)/4 + (ℏ²/2m*) × K_G/2

    其中 κ₁, κ₂ 为主曲率, K_G = κ₁κ₂ 为 Gauss 曲率。

    对球形表面: κ₁ = κ₂ = 1/R, K_G = 1/R²
        V_geo = -(ℏ²/2m*) × 1/(2R²) + (ℏ²/2m*) × 1/(2R²) = 0

    对圆柱面: κ₁ = 1/R, κ₂ = 0, K_G = 0
        V_geo = -(ℏ²/2m*) × 1/(4R²)

    简化模型: 使用 Gauss 曲率 K 的平均值估计修正。

    借鉴 1209_brady-flinchum 的曲率计算方法。

    Parameters
    ----------
    curvature_K : float
        平均 Gauss 曲率 (nm⁻²)
    params : BHZParameters
    h_grid : float
        格点间距 (nm)

    Returns
    -------
    V_geo : float
        几何势修正 (eV)
    """
    # 有效质量 (从 BHZ 参数估计)
    # m* = ℏ²/(2|B|), 其中 B 的单位是 eV·nm²
    hbar = 0.6582  # eV·fs
    # 转换为 eV·nm²·fs²/nm² 单位... 简化:
    # B 直接给出有效质量: |B| = ℏ²/(2m*)
    # 所以 ℏ²/(2m*) = |B|
    hbar2_2m = abs(params.B)  # eV·nm²

    # 几何势: V_geo ≈ -ℏ²/(8m*) × (mean_curvature²)
    # 简化: 使用 K ≈ κ_mean² 估计
    kappa_mean = np.sqrt(abs(curvature_K)) if curvature_K > 0 else 0.0

    V_geo = -hbar2_2m * kappa_mean ** 2 / 4.0

    # 格点修正
    lattice_correction = hbar2_2m * (np.pi / h_grid) ** 2 * 1e-4

    return V_geo + lattice_correction
