"""
stability.py — 数值稳定性分析与共振分岔 (PROJECT_232)

融合种子项目:
  - 1085_PierceRyan: 时延系统的边界碰撞分岔
    (border-collision bifurcations in driven time-delay systems)

核心物理:
  散射振幅的数值计算涉及多个不稳定因素:
    1. 有限差分的高阶格式放大舍入误差
    2. 分波展开的截断误差
    3. 数值积分的网格依赖性
    4. 耦合通道方程的刚度

  本模块分析:
    1. 有限差分的条件数随阶数的变化
    2. Richardson 外推的稳定性
    3. 共振参数的分岔行为 (类似边界碰撞分岔)
    4. 耦合通道系统的 Lyapunov 稳定性

  边界碰撞分岔 (Border-collision Bifurcation):
    在散射共振参数空间中，当共振位置穿越阈值时，
    散射振幅发生不连续变化 (类似 1085_PierceRyan 中的分岔)。

    时延系统: ẋ(t) = -x(t) + b·sgn(x(t-τ)) + F(t)
    其中 F(t) = b·sgn(sgn(1/2 - t mod 1))

    散射类比: 相移 δ(E) 在 E = E_res 处的快速变化
"""
import numpy as np
from constants import EPS_MACH, PI, TWOPI


# ===================================================================
# 有限差分稳定性分析
# ===================================================================

def fd_condition_number(deriv_order, accuracy_order, h):
    """
    有限差分算子的条件数

    对于 m 阶导数的 p 阶精度差分:
      D_h f = Σ w_j f(x_j) / h^m

    条件数:
      κ(D_h) = Σ |w_j| / h^m

    这决定了舍入误差的放大因子:
      ε_output ≈ κ(D_h) · ε_mach · ||f||

    最优步长:
      h_opt ≈ (ε_mach · C / |f^{(m+p)}|)^{1/p}
    使得总误差 (截断 + 舍入) 最小。

    Parameters
    ----------
    deriv_order : int
        导数阶数 m
    accuracy_order : int
        精度阶数 p
    h : float
        步长

    Returns
    -------
    condition_number : float
        差分算子条件数
    roundoff_amplification : float
        舍入误差放大因子
    """
    from finite_difference import central_diff_weights
    offsets, weights = central_diff_weights(deriv_order, accuracy_order)

    cond = np.sum(np.abs(weights)) / (h ** deriv_order)
    roundoff = cond * EPS_MACH

    return cond, roundoff


def optimal_step_size(deriv_order, accuracy_order, f_max_deriv, eps=EPS_MACH):
    """
    计算最优步长

    总误差: E(h) = C_T h^p + C_R ε / h^m
    最小化: dE/dh = 0 → h_opt = (m C_R ε / (p C_T))^{1/(p+m)}

    简化 (设 C_T ≈ C_R ≈ |f^{(m+p)}|):
      h_opt ≈ (ε / |f^{(m+p)}|)^{1/p}

    Parameters
    ----------
    deriv_order : int
        导数阶数 m
    accuracy_order : int
        精度阶数 p
    f_max_deriv : float
        最高阶导数的估计 |f^{(m+p)}|
    eps : float
        机器精度

    Returns
    -------
    h_opt : float
        最优步长
    E_min : float
        最小总误差
    """
    m = deriv_order
    p = accuracy_order
    if abs(f_max_deriv) < EPS_MACH:
        return 1.0, eps

    h_opt = (eps / abs(f_max_deriv)) ** (1.0 / p)
    E_min = abs(f_max_deriv) * h_opt ** p + eps / h_opt ** m
    return h_opt, E_min


def stability_map_fd(deriv_order, h_range, accuracy_orders=None):
    """
    有限差分稳定性图

    对于给定的导数阶数和步长范围，
    计算不同精度阶数下的条件数和误差。

    Parameters
    ----------
    deriv_order : int
        导数阶数
    h_range : ndarray
        步长范围
    accuracy_orders : list of int
        精度阶数列表

    Returns
    -------
    stability_data : dict
        {accuracy_order: {'condition': ndarray, 'error': ndarray}}
    """
    if accuracy_orders is None:
        accuracy_orders = [2, 4, 6, 8]

    stability_data = {}
    for p in accuracy_orders:
        conditions = np.zeros(len(h_range))
        errors = np.zeros(len(h_range))
        for i, h in enumerate(h_range):
            cond, roundoff = fd_condition_number(deriv_order, p, h)
            conditions[i] = cond
            errors[i] = roundoff
        stability_data[p] = {
            'condition': conditions,
            'error': errors
        }
    return stability_data


# ===================================================================
# 共振分岔分析 (融合 1085_PierceRyan)
# ===================================================================

class BorderCollisionMap:
    """
    边界碰撞分岔映射

    借鉴 1085_PierceRyan 中的时延系统:
      ẋ(t) = -x(t) + b·sgn(x(t-τ)) + F(t)

    在散射中的类比:
      定义映射 T: (E_res, Γ_res) → δ_l(E)
      当 E_res 穿越阈值 E_thr 时，相移发生不连续跳变。

    迭代映射:
      x_{n+1} = a·x_n + b·sgn(x_{n-d}) + F_n
    其中 d = τ/Δt 为时延步数。

    分岔参数:
      - a: 衰减率 (与共振宽度相关)
      - b: 反馈强度 (与耦合常数相关)
      - τ: 时延 (与共振寿命相关)

    Attributes
    ----------
    tau : float
        时延参数
    b_force : float
        驱动力强度
    """

    def __init__(self, tau=0.95, b_force=1.35):
        self.tau = tau
        self.b_force = b_force

    def iterate_map(self, Z, x, t, b=None):
        """
        迭代映射一步

        来自 1085_PierceRyan/sample_code.py:
          寻找下一个事件时间 (过零/反馈变化/驱动力变化)

        Parameters
        ----------
        Z : list
            历史过零时间列表
        x : float
            当前状态
        t : float
            当前时间
        b : float, optional
            力强度

        Returns
        -------
        xnew : float
            新状态
        dt : float
            到下一事件的时间
        """
        if b is None:
            b = self.b_force

        s = t % 1.0
        sign = 1 if s < 0.5 else -1
        forcing = b * np.copysign(1.0, sign)
        feedback = np.copysign(1.0, x) * ((-1) ** len(Z)) if x != 0 else 0
        drive = feedback + forcing

        t_d = 0.5 - (t % 0.5)
        t_h = self.tau + Z[0] if len(Z) > 0 else np.inf
        t_z = -x / drive if drive != 0 and (-x / drive) > 0 else np.inf

        dt = min(t_d, t_h, t_z)
        at = np.argmin([t_d, t_h, t_z])

        for i in range(len(Z)):
            Z[i] = Z[i] - dt

        xnew = x + dt * drive if at != 2 else np.copysign(0, -x)

        if at == 2:
            Z.append(0.0)
        if at == 1 and len(Z) > 0:
            del Z[0]

        return xnew, dt

    def simulate(self, tau, Z_init, x0, tmax, b=None):
        """
        模拟时延系统

        Parameters
        ----------
        tau : float
            时延参数
        Z_init : list
            初始历史
        x0 : float
            初始条件
        tmax : float
            最大时间
        b : float, optional
            力强度

        Returns
        -------
        X : list
            位置时间序列
        T : list
            时间序列
        """
        self.tau = tau
        T = [0.0]
        X = [x0]
        Z = list(Z_init)

        max_steps = 100000
        step = 0
        while T[-1] < tmax and step < max_steps:
            xnew, dt = self.iterate_map(Z, X[-1], T[-1], b)
            tnew = T[-1] + dt
            if dt < EPS_MACH:
                break
            T.append(tnew)
            X.append(xnew)
            step += 1

        return X, T


def resonance_bifurcation_analysis(energy_grid, phase_shift_grid,
                                     threshold_energy):
    """
    共振参数穿越阈值的分岔分析

    当共振能量 E_res 接近阈值 E_thr 时:
      - 若 E_res > E_thr: 正常共振 (Breit-Wigner)
      - 若 E_res < E_thr: 虚拟态或束缚态
      - 若 E_res = E_thr: 分岔点 (散射长度发散)

    在 E_res = E_thr 处的行为:
      a₀ → ±∞  (散射长度发散)
      δ₀(E_thr) = π/2  (相移通过 π/2)

    这类似于 1085_PierceRyan 中的边界碰撞分岔:
      系统的定性行为在参数穿越临界值时发生突变。

    Parameters
    ----------
    energy_grid : ndarray
        能量网格
    phase_shift_grid : ndarray
        相移数据
    threshold_energy : float
        阈值能量

    Returns
    -------
    bifurcation_type : str
        分岔类型 ('resonance', 'virtual', 'bound', 'threshold')
    scattering_length : float
        散射长度
    effective_range : float
        有效力程
    """
    # 在阈值附近拟合
    near_thr = np.abs(energy_grid - threshold_energy) < 0.1
    if np.sum(near_thr) < 3:
        return 'unknown', 0.0, 0.0

    E_near = energy_grid[near_thr]
    delta_near = phase_shift_grid[near_thr]

    # 检查相移是否通过 π/2
    max_delta = np.max(np.abs(delta_near))
    if max_delta > PI / 4:
        # 相移大幅变化 → 共振
        bifurcation_type = 'resonance'
    else:
        # 相移变化小
        if delta_near[0] > 0:
            bifurcation_type = 'virtual'
        else:
            bifurcation_type = 'bound'

    # 散射长度估计
    k_near = np.sqrt(np.maximum(2.0 * (E_near - threshold_energy), 0.0))
    valid = k_near > 1e-6
    if np.sum(valid) >= 2:
        k_v = k_near[valid]
        delta_v = delta_near[valid]
        kcot = k_v / np.tan(delta_v + 0j)
        # 线性拟合
        A_mat = np.column_stack([np.ones_like(k_v), k_v ** 2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, np.real(kcot), rcond=None)
            scattering_length = -1.0 / coeffs[0] if abs(coeffs[0]) > EPS_MACH else np.inf
            effective_range = 2.0 * coeffs[1]
        except np.linalg.LinAlgError:
            scattering_length = 0.0
            effective_range = 0.0
    else:
        scattering_length = 0.0
        effective_range = 0.0

    return bifurcation_type, scattering_length, effective_range


# ===================================================================
# 耦合通道稳定性
# ===================================================================

def coupled_channel_stiffness(K_matrix, rho_vec):
    """
    耦合通道方程的刚度分析

    耦合通道 LS 方程:
      T = V + V G₀ T

    刚度取决于:
      - K 矩阵的本征值分布
      - 相空间因子的比值
      - 通道间的耦合强度

    刚度比 = max|λ| / min|λ|

    高刚度 → 需要隐式积分方法

    Parameters
    ----------
    K_matrix : ndarray
        K 矩阵
    rho_vec : ndarray
        相空间因子

    Returns
    -------
    stiffness_ratio : float
        刚度比
    eigenvalues : ndarray
        耦合矩阵本征值
    is_stiff : bool
        是否为刚性系统
    """
    sqrt_rho = np.sqrt(np.maximum(rho_vec, EPS_MACH))
    sqrt_rho_mat = np.diag(sqrt_rho)
    K_tilde = sqrt_rho_mat @ K_matrix @ sqrt_rho_mat

    eigenvalues = np.linalg.eigvals(K_tilde)
    abs_eigs = np.abs(eigenvalues)
    max_eig = np.max(abs_eigs) if len(abs_eigs) > 0 else 0.0
    min_eig = np.min(abs_eigs) if len(abs_eigs) > 0 else EPS_MACH

    stiffness_ratio = max_eig / max(min_eig, EPS_MACH)
    is_stiff = stiffness_ratio > 1000.0

    return stiffness_ratio, eigenvalues, is_stiff


def lyapunov_exponent_estimate(time_series, dt, embedding_dim=3):
    """
    时间序列的最大 Lyapunov 指数估计

    用于判断散射振幅随参数变化的混沌性。

    方法: Rosenstein 算法
      1. 重构相空间: X_i = [x(i), x(i+τ), ..., x(i+(m-1)τ)]
      2. 对每个参考点找最近邻
      3. 跟踪最近邻距离的平均指数增长:
         d(j) ≈ d(0) · e^{λj}
      4. λ = 斜率 of ln d(j) vs j

    Parameters
    ----------
    time_series : ndarray
        时间序列
    dt : float
        采样间隔
    embedding_dim : int
        嵌入维数

    Returns
    -------
    lyap_exp : float
        最大 Lyapunov 指数估计
    is_chaotic : bool
        是否可能为混沌 (λ > 0)
    """
    N = len(time_series)
    if N < embedding_dim + 10:
        return 0.0, False

    # 相空间重构
    tau_delay = max(1, N // (embedding_dim * 10))
    n_vectors = N - (embedding_dim - 1) * tau_delay
    if n_vectors < 10:
        return 0.0, False

    X = np.zeros((n_vectors, embedding_dim))
    for i in range(n_vectors):
        for j in range(embedding_dim):
            X[i, j] = time_series[i + j * tau_delay]

    # 找最近邻 (排除自身和邻近点)
    min_dist = np.inf
    nn_idx = np.zeros(n_vectors, dtype=int)
    for i in range(n_vectors):
        for j in range(n_vectors):
            if abs(i - j) < 3:
                continue
            d = np.linalg.norm(X[i] - X[j])
            if d < min_dist and d > EPS_MACH:
                min_dist = d
                nn_idx[i] = j

    # 跟踪距离演化
    max_iter = min(50, n_vectors // 4)
    if max_iter < 5:
        return 0.0, False

    avg_log_dist = np.zeros(max_iter)
    count = np.zeros(max_iter)

    for i in range(n_vectors - max_iter):
        j = nn_idx[i]
        for k in range(max_iter):
            if i + k >= n_vectors or j + k >= n_vectors:
                break
            d = np.linalg.norm(X[i + k] - X[j + k])
            if d > EPS_MACH:
                avg_log_dist[k] += np.log(d)
                count[k] += 1

    valid = count > 0
    if np.sum(valid) < 5:
        return 0.0, False

    avg_log_dist[valid] /= count[valid]
    k_vals = np.arange(max_iter)[valid]

    # 线性拟合求斜率
    if len(k_vals) >= 2:
        A_mat = np.column_stack([k_vals.astype(float), np.ones(len(k_vals))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, avg_log_dist[valid], rcond=None)
            lyap_exp = coeffs[0] / dt
        except np.linalg.LinAlgError:
            lyap_exp = 0.0
    else:
        lyap_exp = 0.0

    is_chaotic = lyap_exp > 0.01

    return lyap_exp, is_chaotic
