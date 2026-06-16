"""
kinematics.py — 散射运动学与质心系网格生成 (PROJECT_232)

融合种子项目:
  - 880_polar_ode: 极坐标参数化 (polar_parameters, polar_exact, polar_deriv)
  - 492_gridlines: 结构化网格生成 (grid_rectangular, grid_polar, grid_triangular)
  - 1052_mctools_cdft_ncrystal: FFT频率网格 (getOmegaFromTime)

核心功能:
  1. 质心系四动量构造
  2. 质心能量 √s 网格 (线性 / 对数 / 切比雪夫)
  3. 散射角 θ* 的极坐标网格
  4. Mandelstam 变量 (s, t, u) 的完整运动学映射
  5. 相空间因子 ρ(s) 与速度因子 β(s)

物理背景:
  对于 2→2 散射过程 a + b → c + d，
  在质心系中入射与出射三动量大小分别为:
    |p⃗_i*| = λ^{1/2}(s, m_a², m_b²) / (2√s)
    |p⃗_f*| = λ^{1/2}(s, m_c², m_d²) / (2√s)
  其中 λ 为 Källén 三角函数:
    λ(x,y,z) = x² + y² + z² − 2xy − 2xz − 2yz
"""
import numpy as np
from constants import (
    cm_momentum, mandelstam_s, mandelstam_t, mandelstam_u,
    velocity_factor, PI, TWOPI, FOURPI, EPS_MACH, MASS_PI_PLUS
)


# ---------------------------------------------------------------------------
# 质心能量网格生成 (融合 492_gridlines 的多种网格类型)
# ---------------------------------------------------------------------------

def grid_linear(s_min, s_max, n_pts):
    """
    线性等距 √s 网格

    Parameters
    ----------
    s_min, s_max : float
        √s 的范围 [GeV]
    n_pts : int
        网格点数

    Returns
    -------
    sqrt_s_grid : ndarray, shape (n_pts,)
    s_grid : ndarray, shape (n_pts,)
        s = (√s)²
    ds : float
        网格间距 Δ(√s)
    """
    if n_pts < 2:
        raise ValueError("grid_linear: n_pts must be >= 2")
    if s_max <= s_min:
        raise ValueError("grid_linear: s_max must be > s_min")
    sqrt_s_grid = np.linspace(s_min, s_max, n_pts)
    s_grid = sqrt_s_grid ** 2
    ds = (s_max - s_min) / (n_pts - 1)
    return sqrt_s_grid, s_grid, ds


def grid_chebyshev(s_min, s_max, n_pts):
    """
    切比雪夫节点网格 (在区间 [s_min, s_max] 上)

    切比雪夫节点在端点附近密集，适合多项式插值与高阶差分:
      √s_k = (s_min + s_max)/2 + (s_max - s_min)/2 · cos(π(2k-1)/(2N))
      k = 1, 2, ..., N

    这种分布在阈值 √s ≈ m_a + m_b 附近提供更高的分辨率，
    对于散射振幅在阈值处的解析结构（如 branch cut）至关重要。

    Parameters
    ----------
    s_min, s_max : float
        √s 范围 [GeV]
    n_pts : int
        网格点数

    Returns
    -------
    sqrt_s_grid : ndarray, shape (n_pts,)
    s_grid : ndarray, shape (n_pts,)
    weights : ndarray, shape (n_pts,)
        切比雪夫权重 (用于 Gauss-Chebyshev 求积)
    """
    if n_pts < 2:
        raise ValueError("grid_chebyshev: n_pts must be >= 2")
    k = np.arange(1, n_pts + 1)
    theta_k = PI * (2.0 * k - 1.0) / (2.0 * n_pts)
    # 将 [-1,1] 映射到 [s_min, s_max]
    x_k = np.cos(theta_k)
    sqrt_s_grid = 0.5 * (s_max + s_min) + 0.5 * (s_max - s_min) * x_k
    # 排序为升序
    idx = np.argsort(sqrt_s_grid)
    sqrt_s_grid = sqrt_s_grid[idx]
    s_grid = sqrt_s_grid ** 2
    # 切比雪夫求积权重: w_k = (π/N) sin(θ_k)
    weights = (PI / n_pts) * np.sin(theta_k)
    weights = weights[idx]
    return sqrt_s_grid, s_grid, weights


def grid_log_threshold(s_thr, s_max, n_pts, alpha=5.0):
    """
    对数阈值密集网格

    在阈值 s_thr = (m_a + m_b)² 附近使用对数加密:
      √s_k = s_thr + (s_max - s_thr) · (exp(α·k/N) - 1) / (exp(α) - 1)

    参数 α 控制阈值附近的密度。α 越大，阈值附近越密集。

    这对研究散射振幅在阈值处的行为至关重要，因为:
      - S 波散射长度 a₀ 由 k cot δ₀ → -1/a₀ 在 k→0 极限定义
      - 有效范围展开 k cot δ₀ = -1/a₀ + (1/2)r₀k² + ... 在 k 小时最精确

    Parameters
    ----------
    s_thr : float
        阈值 √s = m_a + m_b [GeV]
    s_max : float
        最大 √s [GeV]
    n_pts : int
        网格点数
    alpha : float
        对数压缩参数 (default: 5.0)

    Returns
    -------
    sqrt_s_grid : ndarray
    s_grid : ndarray
    ds_local : ndarray
        局部网格间距 (非均匀)
    """
    if s_max <= s_thr:
        raise ValueError("grid_log_threshold: s_max must be > s_thr")
    if n_pts < 3:
        raise ValueError("grid_log_threshold: n_pts must be >= 3")
    k_idx = np.arange(n_pts, dtype=np.float64)
    frac = (np.exp(alpha * k_idx / (n_pts - 1)) - 1.0) / (np.exp(alpha) - 1.0)
    sqrt_s_grid = s_thr + (s_max - s_thr) * frac
    s_grid = sqrt_s_grid ** 2
    ds_local = np.diff(sqrt_s_grid)
    ds_local = np.append(ds_local, ds_local[-1])
    return sqrt_s_grid, s_grid, ds_local


# ---------------------------------------------------------------------------
# 散射角网格 (融合 880_polar_ode 的极坐标参数化)
# ---------------------------------------------------------------------------

def grid_cos_theta(n_pts, symmetric=True):
    """
    cos θ* 均匀网格

    对于非极化 2→2 散射，微分截面仅依赖于散射角 θ*
    (质心系中入射与出射方向的夹角)。

    cos θ* ∈ [-1, 1] (全角度范围)
    或 cos θ* ∈ [0, 1] (对于全同粒子，利用前-后对称性)

    Parameters
    ----------
    n_pts : int
        角度网格点数
    symmetric : bool
        若 True，利用 cos θ → -cos θ 对称性仅计算 [0,1]

    Returns
    -------
    cos_theta : ndarray, shape (n_pts,)
    theta : ndarray, shape (n_pts,)
        θ* [弧度]
    weights : ndarray, shape (n_pts,)
        Gauss-Legendre 求积权重
    """
    if n_pts < 2:
        raise ValueError("grid_cos_theta: n_pts must be >= 2")
    # Gauss-Legendre 节点与权重
    cos_theta, weights = np.polynomial.legendre.leggauss(n_pts)
    if symmetric:
        # 仅保留 cos θ ≥ 0 的部分
        mask = cos_theta >= 0.0
        cos_theta = cos_theta[mask]
        weights = 2.0 * weights[mask]  # 对称因子
        if len(cos_theta) == 0:
            cos_theta = np.array([0.5])
            weights = np.array([2.0])
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return cos_theta, theta, weights


def polar_grid_2d(n_sqrt_s, n_theta, s_min, s_max,
                  m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS):
    """
    极坐标 (√s, θ*) 二维网格

    融合 492_gridlines/grid_polar 的极坐标网格思想与
    880_polar_ode 的参数化方法。

    在极坐标下，四动量转移 t 表示为:
      t(s, cos θ*) = m_a² + m_c² - 2E_a*E_c* + 2|p⃗_i*||p⃗_f*| cos θ*

    其中 E_a* = (s + m_a² - m_b²)/(2√s)
         E_c* = (s + m_c² - m_d²)/(2√s)

    Parameters
    ----------
    n_sqrt_s : int
        √s 方向网格点数
    n_theta : int
        θ* 方向网格点数
    s_min, s_max : float
        √s 范围
    m_a, m_b : float
        入射粒子质量 (假设 m_c=m_a, m_d=m_b 弹性散射)

    Returns
    -------
    sqrt_s_2d : ndarray, shape (n_sqrt_s, n_theta)
    cos_theta_2d : ndarray, shape (n_sqrt_s, n_theta)
    t_2d : ndarray, shape (n_sqrt_s, n_theta)
        Mandelstam t 变量
    """
    sqrt_s_1d, _, _ = grid_linear(s_min, s_max, n_sqrt_s)
    cos_theta_1d, _, w_theta = grid_cos_theta(n_theta, symmetric=False)

    # 构造二维网格
    sqrt_s_2d, cos_theta_2d = np.meshgrid(sqrt_s_1d, cos_theta_1d, indexing='ij')

    # 弹性散射: m_c = m_a, m_d = m_b
    m_c, m_d = m_a, m_b
    s_2d = sqrt_s_2d ** 2

    # 质心系能量
    E_a_star = (s_2d + m_a ** 2 - m_b ** 2) / (2.0 * sqrt_s_2d)
    E_c_star = (s_2d + m_c ** 2 - m_d ** 2) / (2.0 * sqrt_s_2d)

    # 质心系动量
    p_i_star = np.zeros_like(s_2d)
    p_f_star = np.zeros_like(s_2d)
    above_thr = sqrt_s_2d > (m_a + m_b) + EPS_MACH
    for idx in np.ndindex(s_2d.shape):
        if above_thr[idx]:
            p_i_star[idx] = cm_momentum(s_2d[idx], m_a, m_b)
            p_f_star[idx] = cm_momentum(s_2d[idx], m_c, m_d)

    # Mandelstam t = m_a² + m_c² - 2E_a*E_c* + 2|p_i*||p_f*|cos θ*
    t_2d = m_a ** 2 + m_c ** 2 - 2.0 * E_a_star * E_c_star + 2.0 * p_i_star * p_f_star * cos_theta_2d

    return sqrt_s_2d, cos_theta_2d, t_2d


# ---------------------------------------------------------------------------
# 相空间因子
# ---------------------------------------------------------------------------

def phase_space_factor(s, m_a, m_b):
    """
    两体相空间因子 ρ(s)

    ρ(s) = 2|p⃗*| / √s = √λ(s, m_a², m_b²) / s

    这在散射振幅的幺正性关系中扮演核心角色:
      Im T_l(s) = ρ(s) |T_l(s)|²    (弹性区幺正性)

    其中 T_l 为分波振幅 (与 S 矩阵关系: S_l = 1 + 2iT_l)

    Parameters
    ----------
    s : float or ndarray
        质心能量平方 [GeV²]
    m_a, m_b : float
        粒子质量 [GeV]

    Returns
    -------
    rho : float or ndarray
        相空间因子 [GeV⁻¹]
    """
    s = np.asarray(s, dtype=np.float64)
    rho = np.zeros_like(s)
    above = s > (m_a + m_b) ** 2
    if np.any(above):
        s_above = s[above] if s.ndim > 0 else s
        p_star = cm_momentum(s_above, m_a, m_b)
        rho_val = 2.0 * p_star / np.sqrt(s_above)
        if s.ndim > 0:
            rho[above] = rho_val
        else:
            return rho_val
    return rho


def lorentz_boost(p_fourvec, beta_vec):
    """
    Lorentz .boost 四矢量

    将四动量从实验室系变换到质心系 (或反之)。

    对于 boost 速度 β⃗ = v⃗/c:
      γ = 1/√(1-β²)
      E' = γ(E - β⃗·p⃗)
      p⃗' = p⃗ + [(γ-1)(β⃗·p⃗)/β² - γE] β⃗

    Parameters
    ----------
    p_fourvec : ndarray, shape (..., 4)
        四动量 (E, px, py, pz)
    beta_vec : ndarray, shape (3,)
        boost 速度矢量 (β_x, β_y, β_z)

    Returns
    -------
    p_boosted : ndarray, shape (..., 4)
        boost 后的四动量
    """
    beta_vec = np.asarray(beta_vec, dtype=np.float64)
    beta2 = np.dot(beta_vec, beta_vec)
    if beta2 >= 1.0 - EPS_MACH:
        raise ValueError("lorentz_boost: |β| >= 1, unphysical boost")
    if beta2 < EPS_MACH:
        return p_fourvec.copy()

    gamma = 1.0 / np.sqrt(1.0 - beta2)
    p_fourvec = np.asarray(p_fourvec, dtype=np.float64)
    E = p_fourvec[..., 0]
    p3 = p_fourvec[..., 1:4]

    beta_dot_p = np.dot(p3, beta_vec) if p3.ndim == 1 else p3 @ beta_vec
    factor = (gamma - 1.0) / beta2

    E_new = gamma * (E - beta_dot_p)
    p3_new = p3 + (factor * beta_dot_p - gamma * E)[..., np.newaxis] * beta_vec

    result = np.zeros_like(p_fourvec)
    result[..., 0] = E_new
    result[..., 1:4] = p3_new
    return result


def cm_frame_four_momenta(sqrt_s, m_a, m_b, cos_theta):
    """
    构造质心系中 2→2 散射的四个四动量

    约定:
      p1 = (E_a*, 0, 0, +|p_i*|)           入射粒子 a
      p2 = (E_b*, 0, 0, -|p_i*|)           入射粒子 b
      p3 = (E_c*, |p_f*|sin θ*, 0, |p_f*|cos θ*)  出射粒子 c
      p4 = (E_d*, -|p_f*|sin θ*, 0, -|p_f*|cos θ*) 出射粒子 d

    Parameters
    ----------
    sqrt_s : float
        质心能量 [GeV]
    m_a, m_b : float
        入射粒子质量 [GeV] (弹性散射: m_c=m_a, m_d=m_b)
    cos_theta : float
        质心系散射角余弦

    Returns
    -------
    p1, p2, p3, p4 : ndarray, shape (4,)
        四个四动量
    """
    if sqrt_s < m_a + m_b - EPS_MACH:
        raise ValueError(
            f"cm_frame_four_momenta: √s={sqrt_s:.4f} < threshold {m_a + m_b:.4f}")
    s = sqrt_s ** 2
    m_c, m_d = m_a, m_b  # 弹性散射

    E_a = (s + m_a ** 2 - m_b ** 2) / (2.0 * sqrt_s)
    E_b = (s + m_b ** 2 - m_a ** 2) / (2.0 * sqrt_s)
    E_c = (s + m_c ** 2 - m_d ** 2) / (2.0 * sqrt_s)
    E_d = (s + m_d ** 2 - m_c ** 2) / (2.0 * sqrt_s)

    p_i = cm_momentum(s, m_a, m_b)
    p_f = cm_momentum(s, m_c, m_d)

    sin_theta = np.sqrt(max(1.0 - cos_theta ** 2, 0.0))

    p1 = np.array([E_a, 0.0, 0.0, p_i])
    p2 = np.array([E_b, 0.0, 0.0, -p_i])
    p3 = np.array([E_c, p_f * sin_theta, 0.0, p_f * cos_theta])
    p4 = np.array([E_d, -p_f * sin_theta, 0.0, -p_f * cos_theta])

    return p1, p2, p3, p4


def cross_check_mandelstam(p1, p2, p3, p4, m_a, m_b):
    """
    验证 Mandelstam 关系 s + t + u = Σ m_i²

    Parameters
    ----------
    p1, p2, p3, p4 : ndarray, shape (4,)
        四动量
    m_a, m_b : float
        粒子质量

    Returns
    -------
    s, t, u : float
        Mandelstam 变量
    check_ok : bool
        s + t + u ≈ Σm_i² 是否成立
    """
    s = mandelstam_s(p1, p2)
    t = mandelstam_t(p1, p3)
    u = mandelstam_u(p1, p4)
    sum_m2 = m_a ** 2 + m_b ** 2 + m_a ** 2 + m_b ** 2
    stu_sum = s + t + u
    rel_err = abs(stu_sum - sum_m2) / max(abs(sum_m2), EPS_MACH)
    return s, t, u, rel_err < 1e-8
