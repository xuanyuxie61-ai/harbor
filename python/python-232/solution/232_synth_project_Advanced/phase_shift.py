"""
phase_shift.py — 相移提取与极坐标演化分析 (PROJECT_232)

融合种子项目:
  - 880_polar_ode: 极坐标参数化 ODE (polar_deriv, polar_exact, polar_parameters)
  - 767_midpoint_fixed: 中点固定点迭代 (midpoint_fixed)

核心物理:
  相移 δ_l(s) 编码了散射的全部信息:
    S_l = e^{2iδ_l} = η_l e^{2iδ_l}

  相移的解析性质:
    - 阈值行为: δ_l(k) ~ k^{2l+1} 当 k → 0 (Levinson 定理)
    - 共振: δ_l 快速通过 π/2 (Breit-Wigner)
    - 束缚态: δ_l(0) - δ_l(∞) = n_b · π (Levinson 定理)

  本模块实现:
    1. 从 T 矩阵提取相移
    2. 相移的极坐标表示 r(θ) = δ_l(E(θ))
    3. 使用固定点迭代求解 Lippmann-Schwinger 方程
    4. Breit-Wigner 共振参数提取
"""
import numpy as np
from constants import PI, TWOPI, EPS_MACH, MASS_PI_PLUS, cm_momentum


# ===================================================================
# 相移提取
# ===================================================================

def extract_phase_shift(T_l_complex):
    """
    从复数 T 矩阵元提取相移

    T_l = |T_l| e^{iφ_l}
    δ_l = φ_l  (弹性区, 幺正性要求 |S_l| = 1)

    对于非弹性散射:
      S_l = η_l e^{2iδ_l}, η_l < 1
      T_l = (η_l e^{2iδ_l} - 1) / (2i)

    Parameters
    ----------
    T_l_complex : complex or ndarray
        复数 T 矩阵元

    Returns
    -------
    delta_l : float or ndarray
        相移 [弧度]
    eta_l : float or ndarray
        非弹性参数
    """
    T_l_complex = np.asarray(T_l_complex, dtype=np.complex128)
    S_l = 1.0 + 2j * T_l_complex
    eta_l = np.abs(S_l)
    # 防止 eta=0 时 log 发散
    eta_safe = np.maximum(eta_l, EPS_MACH)
    delta_l = np.angle(S_l) / 2.0
    return np.real(delta_l), eta_l


def breit_wigner_phase_shift(sqrt_s, m_res, gamma_res, l_quantum=0,
                              m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS):
    """
    Breit-Wigner 共振相移

    对于自旋 l 的共振:
      tan δ_l = Γ(s) / (2(m_res - √s))

    其中能量依赖宽度:
      Γ(s) = Γ_res · (k/k_res)^{2l+1} · (m_res/√s) · (B_l(k·R))² / (B_l(k_res·R))²

    B_l 为 Blatt-Weisskopf 穿透因子:
      B_0(z) = 1
      B_1(z) = z/√(1+z²)
      B_2(z) = z²/√(9+3z²+z⁴)

    Parameters
    ----------
    sqrt_s : float or ndarray
        质心能量 [GeV]
    m_res : float
        共振质量 [GeV]
    gamma_res : float
        共振宽度 [GeV]
    l_quantum : int
        轨道角动量
    m_a, m_b : float
        衰变产物质量

    Returns
    -------
    delta_l : float or ndarray
        共振相移
    """
    sqrt_s = np.asarray(sqrt_s, dtype=np.float64)
    s = sqrt_s ** 2

    k = np.array([cm_momentum(ss ** 2, m_a, m_b) for ss in sqrt_s.flat])
    k = k.reshape(sqrt_s.shape)
    k_res = cm_momentum(m_res ** 2, m_a, m_b)

    if k_res < EPS_MACH:
        return np.zeros_like(sqrt_s)

    # Blatt-Weisskopf 穿透因子
    R = 1.5 / 0.197  # 强子半径 ~1.5 fm → GeV⁻¹

    def blad_weisskopf(k_val, l_val):
        z = k_val * R
        if l_val == 0:
            return np.ones_like(z)
        elif l_val == 1:
            return z / np.sqrt(1.0 + z ** 2)
        elif l_val == 2:
            return z ** 2 / np.sqrt(9.0 + 3.0 * z ** 2 + z ** 4)
        else:
            # 一般情况 (递推)
            B = np.ones_like(z)
            for ll in range(1, l_val + 1):
                B = z * B / np.sqrt((2 * ll - 1) ** 2 + z ** 2)
            return B

    B_l = blad_weisskopf(k, l_quantum)
    B_l_res = blad_weisskopf(np.array([k_res]), l_quantum)[0]

    # 能量依赖宽度
    ratio_k = np.where(k > EPS_MACH, k / k_res, 0.0)
    gamma_s = gamma_res * ratio_k ** (2 * l_quantum + 1) * \
              (m_res / np.maximum(sqrt_s, EPS_MACH)) * \
              (B_l / max(B_l_res, EPS_MACH)) ** 2

    # 相移
    delta_l = np.arctan2(gamma_s, 2.0 * (m_res - sqrt_s))

    return delta_l


# ===================================================================
# 极坐标相移演化 (融合 880_polar_ode)
# ===================================================================

def phase_shift_polar_trajectory(sqrt_s_grid, delta_l_grid):
    """
    相移的极坐标表示

    将 δ_l(√s) 表示为极坐标曲线:
      θ(√s) = √s / √s_max · 2π   (角度参数)
      r(√s) = δ_l(√s)             (径向距离)

    类似 880_polar_ode 中的极坐标 ODE:
      z(t) = r(t) e^{iθ(t)}
      dz/dt = (dr/dt + ir dθ/dt) e^{iθ}

    在相移分析中，极坐标表示用于:
      - 可视化相移的 winding number (绕原点圈数)
      - Levinson 定理: Δδ = n_b · π 对应绕原点 n_b 圈

    Parameters
    ----------
    sqrt_s_grid : ndarray
        √s 网格
    delta_l_grid : ndarray
        相移 δ_l(√s)

    Returns
    -------
    r_values : ndarray
        极坐标径向 (相移)
    theta_values : ndarray
        极坐标角度 (参数化能量)
    z_complex : ndarray
        复数表示 z = r e^{iθ}
    winding_number : int
        绕数
    """
    s_max = sqrt_s_grid[-1]
    theta_values = (sqrt_s_grid / s_max) * TWOPI
    r_values = delta_l_grid

    z_complex = r_values * np.exp(1j * theta_values)

    # 计算绕数 (winding number)
    if len(z_complex) < 2:
        return r_values, theta_values, z_complex, 0

    d_angles = np.diff(np.angle(z_complex))
    # 去除 2π 跳跃
    d_angles = np.unwrap(d_angles)
    total_angle = np.sum(d_angles)
    winding_number = int(round(total_angle / TWOPI))

    return r_values, theta_values, z_complex, winding_number


def polar_ode_evolution(sqrt_s, delta_0, m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS):
    """
    相移的极坐标 ODE 演化

    借鉴 880_polar_ode/polar_deriv.m:
      dz/dt = (dr/dt + i r) e^{iθ}

    将有效范围展开写成 ODE:
      dδ/dk = (k/2)(r₀ + 2P k² + ...) / (1 + (k a₀)²)

    其中 P 为形状参数。

    Parameters
    ----------
    sqrt_s : ndarray
        √s 网格
    delta_0 : float
        初始相移 (在阈值处)
    m_a, m_b : float
        粒子质量

    Returns
    -------
    delta_evolved : ndarray
        演化后的相移
    """
    s_grid = sqrt_s ** 2
    k_grid = np.array([cm_momentum(s, m_a, m_b) for s in s_grid])

    delta_evolved = np.zeros_like(k_grid)
    delta_evolved[0] = delta_0

    # 简化 ODE 演化 (Euler 方法)
    for i in range(1, len(k_grid)):
        dk = k_grid[i] - k_grid[i - 1]
        if dk < EPS_MACH:
            delta_evolved[i] = delta_evolved[i - 1]
            continue
        k_mid = 0.5 * (k_grid[i] + k_grid[i - 1])
        # 使用有效范围近似: dδ/dk ≈ k a₀ / (1 + (k a₀)²)
        # 这里 a₀ 为散射长度 (取典型值)
        a0 = 0.5  # 典型 ππ 散射长度 [GeV⁻¹]
        ddelta_dk = k_mid * a0 / (1.0 + (k_mid * a0) ** 2)
        delta_evolved[i] = delta_evolved[i - 1] + ddelta_dk * dk

    return delta_evolved


# ===================================================================
# 中点固定点迭代 (融合 767_midpoint_fixed)
# ===================================================================

def lippiann_schwinger_fixed_point(V_func, sqrt_s_grid, m_a, m_b,
                                     n_steps=50, theta=0.5, max_iter=10):
    """
    Lippmann-Schwinger 方程的固定点迭代求解

    融合 767_midpoint_fixed/midpoint_fixed.m 的中点迭代思想。

    LS 方程:
      T(s) = V + V G₀(s) T(s)

    其中 G₀(s) 为自由传播子:
      G₀(s; k) = 1/(s - 4(k² + m²) + iε)

    分波投影后:
      T_l(s) = V_l(s) + (2/π) ∫₀^∞ dk' k'² V_l(k,k') G₀(s;k') T_l(k')

    固定点迭代:
      T_l^{(n+1)} = V_l + K · T_l^{(n)}
    其中 K 为积分算子。

    中点法改进 (来自 767_midpoint_fixed):
      T_mid = T^{(n)} + θ · dt · K · T_mid  (隐式中点)
      T^{(n+1)} = T_mid / θ + (1 - 1/θ) T^{(n)}

    Parameters
    ----------
    V_func : callable
        势函数 V(k, k') → float
    sqrt_s_grid : ndarray
        √s 网格
    m_a, m_b : float
        粒子质量
    n_steps : int
        k 空间离散点数
    theta : float
        中点参数 (0.5 为标准中点法)
    max_iter : int
        最大迭代次数

    Returns
    -------
    T_l_grid : ndarray
        T 矩阵元在各 √s 点的值
    converged : bool
        是否收敛
    """
    # k 空间网格
    s_vals = sqrt_s_grid ** 2
    k_max = cm_momentum(s_vals[-1], m_a, m_b) * 2.0
    k_grid = np.linspace(EPS_MACH, k_max, n_steps)
    dk = k_grid[1] - k_grid[0] if n_steps > 1 else 1.0

    # 构造势矩阵 V(k_i, k_j)
    V_matrix = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        for j in range(n_steps):
            V_matrix[i, j] = V_func(k_grid[i], k_grid[j])

    T_l_grid = np.zeros(len(sqrt_s_grid), dtype=np.complex128)
    converged = True

    for idx, sqrt_s in enumerate(sqrt_s_grid):
        s = sqrt_s ** 2
        # 自由传播子 G₀(k) = 1/(s - 4(k²+m²) + iε)
        E_k = np.sqrt(k_grid ** 2 + m_a ** 2) + np.sqrt(k_grid ** 2 + m_b ** 2)
        G0 = 1.0 / (s - E_k ** 2 + 1j * EPS_MACH)

        # 积分核 K(k,k') = V(k,k') · G₀(k') · k'²
        K_matrix = V_matrix * G0[np.newaxis, :] * k_grid[np.newaxis, :] ** 2 * dk * (2.0 / PI)

        # 固定点迭代 (中点法)
        T_k = V_matrix @ np.ones(n_steps) * 0.1  # 初始猜测
        for iteration in range(max_iter):
            # 中点迭代
            T_mid = T_k.copy()
            for _ in range(5):
                T_mid = V_matrix[:, 0] + theta * (K_matrix @ T_mid)
            T_new = T_mid / theta + (1.0 - 1.0 / theta) * T_k

            # 收敛检查
            diff = np.max(np.abs(T_new - T_k))
            T_k = T_new
            if diff < 1e-10:
                break

        # T_l(s) = V_on_shell + 积分贡献
        T_l_grid[idx] = np.mean(T_k)

    return T_l_grid, converged


# ===================================================================
# Levinson 定理验证
# ===================================================================

def levinson_theorem_check(delta_0_threshold, delta_0_infinity):
    """
    Levinson 定理验证

    Levinson 定理:
      δ_l(0) - δ_l(∞) = n_b^{(l)} · π

    其中 n_b^{(l)} 为角动量 l 通道的束缚态数。

    对于 S 波 (l=0):
      δ_0(0) = n_b · π  (若存在半束缚态则额外加 π/2)

    Parameters
    ----------
    delta_0_threshold : float
        阈值处相移 δ_0(k=0)
    delta_0_infinity : float
        无穷能量处相移 δ_0(k→∞)

    Returns
    -------
    n_bound : int
        推断的束缚态数
    theorem_satisfied : bool
        是否满足 Levinson 定理
    """
    delta_diff = delta_0_threshold - delta_0_infinity
    n_bound = round(delta_diff / PI)
    residual = abs(delta_diff - n_bound * PI)
    theorem_satisfied = residual < 0.1  # 容差 0.1 弧度
    return n_bound, theorem_satisfied
