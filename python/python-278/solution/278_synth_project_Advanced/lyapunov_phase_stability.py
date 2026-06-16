"""
lyapunov_phase_stability.py
============================
基于最大 Lyapunov 指数 (LLE) 的 CALPHAD 相稳定性分析。

种子项目 1204_velichko-andrei_lle-chaos-demos:
  - 使用 KNN 预测误差曲线估计 LLE
  - Rosenstein 方法
  - 自相关时间估计

映射到相稳定性:
  对于 Cahn-Hilliard 方程的时间序列 c(x,t),
  将其视为动力系统:
    dc/dt = F[c]

  最大 Lyapunov 指数衡量相邻轨道的分离速率:
    ||delta c(t)|| ~ ||delta c(0)|| * exp(LLE * t)

  LLE > 0: 轨道指数分离 → spinodal 分解 (不稳定)
  LLE = 0: 中性稳定性 → spinodal 边界
  LLE < 0: 轨道收敛 → 稳定相

计算方法:
  1. 从 CH 模拟中采样状态序列 {c_n}
  2. 构建延迟嵌入 (Takens 定理):
     X_n = [c_n, c_{n+tau}, ..., c_{n+(m-1)*tau}]
  3. 使用 KNN 方法估计 LLE:
     找最近邻对, 追踪距离随时间的演化
  4. LLE = d/dt <ln ||delta X(t)||>

同时实现自由能 Lyapunov 泛函:
  V[c] = integral [G(c) + (kappa/2)|nabla c|^2] dx
  dV/dt <= 0 (热力学第二定律)
"""

import numpy as np
from calphad_fec_constants import SA_N_WAVENUM, SA_OMEGA_MAX


def lyapunov_knn_estimate(time_series, embed_dim=5, time_lag=1,
                          n_neighbors=5, max_horizon=20):
    """
    使用 KNN 方法估计最大 Lyapunov 指数。

    算法 (Rosenstein et al. 1993):
    1. 构建延迟嵌入:
       X_i = [s(i), s(i+tau), ..., s(i+(m-1)*tau)]
    2. 对每个点 X_i, 找 K 个最近邻 (排除时间近邻)
    3. 追踪每对最近邻的距离随预测步数 k 的演化:
       d_i(k) = ||X_{i+k} - X_{nn(i)+k}||
    4. 平均: <d(k)> = (1/N) sum_i d_i(k)
    5. LLE = 斜率 of ln <d(k)> vs k

    Parameters
    ----------
    time_series : np.ndarray
        标量时间序列 (如某点浓度随时间的演化)
    embed_dim : int
        嵌入维度 m
    time_lag : int
        延迟时间 tau
    n_neighbors : int
        近邻数 K
    max_horizon : int
        最大预测步数

    Returns
    -------
    dict
        {
            'lle': float,             # 最大 Lyapunov 指数
            'horizons': np.ndarray,   # 预测步数
            'mean_distances': np.ndarray,  # 平均距离
            'is_unstable': bool,      # 是否不稳定
        }
    """
    s = np.asarray(time_series, dtype=float)
    N = len(s)
    m = embed_dim
    tau = time_lag

    # 延迟嵌入
    n_embed = N - (m - 1) * tau
    if n_embed < 2 * max_horizon:
        # 数据太短, 返回零
        return {
            'lle': 0.0,
            'horizons': np.arange(max_horizon),
            'mean_distances': np.ones(max_horizon) * 1e-10,
            'is_unstable': False,
        }

    X = np.zeros((n_embed, m))
    for i in range(n_embed):
        for j in range(m):
            X[i, j] = s[i + j * tau]

    # 找最近邻 (排除时间近邻, 避免自相关偏差)
    min_time_sep = max(m * tau, 2)
    K = min(n_neighbors, n_embed - 2 * max_horizon - 1)
    K = max(K, 1)

    # 平均距离曲线
    mean_dist = np.zeros(max_horizon)
    count = np.zeros(max_horizon)

    for i in range(n_embed - max_horizon):
        # 计算到所有非近邻点的距离
        dists = np.full(n_embed, np.inf)
        for j in range(n_embed):
            if abs(i - j) < min_time_sep:
                continue
            dists[j] = np.linalg.norm(X[i] - X[j])

        # 找 K 个最近邻
        if K >= n_embed:
            continue
        nn_idx = np.argpartition(dists, K)[:K]

        for k in range(max_horizon):
            for nn in nn_idx:
                if i + k < n_embed and nn + k < n_embed:
                    d = np.linalg.norm(X[i + k] - X[nn + k])
                    d = max(d, 1e-30)
                    mean_dist[k] += np.log(d)
                    count[k] += 1

    # 平均
    valid = count > 0
    mean_dist[valid] /= count[valid]
    mean_dist[~valid] = mean_dist[np.where(valid)[0][0]] if np.any(valid) else 0.0

    # LLE = 线性回归的斜率
    horizons = np.arange(max_horizon)
    if np.sum(valid) > 2:
        v_h = horizons[valid]
        v_d = mean_dist[valid]
        # 最小二乘
        A = np.vstack([v_h, np.ones(len(v_h))]).T
        try:
            result = np.linalg.lstsq(A, v_d, rcond=None)
            lle = result[0][0]
        except np.linalg.LinAlgError:
            lle = 0.0
    else:
        lle = 0.0

    return {
        'lle': float(lle),
        'horizons': horizons,
        'mean_distances': mean_dist,
        'is_unstable': lle > 1e-6,
    }


def linear_stability_dispersion(T, x0, phase, kappa, M0,
                                n_wavenum=None, omega_max=None):
    """
    Cahn-Hilliard 方程的线性稳定性色散关系分析。

    对均匀态 c = x0 + epsilon * exp(ikx + omega*t),
    线性化得到色散关系:

    omega(k) = -M * k² * [d²G/dc²|_{x0} + kappa * k²]

    稳定性条件:
    - 如果 d²G/dc²|_{x0} > 0: omega(k) < 0 对所有 k → 稳定
    - 如果 d²G/dc²|_{x0} < 0: omega(k) > 0 对某些 k → 不稳定
      (spinodal 区域)

    最不稳定波数:
    k_max = sqrt(-d²G/dc² / (2*kappa))
    omega_max_growth = M * (d²G/dc²)² / (4*kappa)

    Parameters
    ----------
    T : float
        温度 (K)
    x0 : float
        均匀态成分
    phase : str
        相名称
    kappa : float
        梯度能系数
    M0 : float
        基准迁移率
    n_wavenum : int
        波数采样点数
    omega_max : float
        最大角频率

    Returns
    -------
    dict
        {
            'wavenumbers': np.ndarray,
            'growth_rates': np.ndarray,
            'k_critical': float,      # 临界波数 (omega=0)
            'k_max_growth': float,    # 最快增长波数
            'omega_max': float,       # 最大增长率
            'is_spinodal': bool,      # 是否处于 spinodal 区
        }
    """
    from gibbs_energy_calphad import second_derivative_G, gibbs_substitutional

    if n_wavenum is None:
        n_wavenum = SA_N_WAVENUM
    if omega_max is None:
        omega_max = SA_OMEGA_MAX

    d2G = second_derivative_G(x0, T, phase)

    # 迁移率
    Q_act = 140000.0
    R = 8.3145
    x0_safe = np.clip(x0, 1e-10, 1.0 - 1e-10)
    M = M0 * x0_safe * (1.0 - x0_safe) * np.exp(-Q_act / (R * T))
    M = max(M, 1e-40)

    # 波数网格 (对数均匀)
    k = np.logspace(np.log10(1e3), np.log10(omega_max ** 0.5), n_wavenum)

    # 色散关系: omega(k) = -M * k^2 * (d2G + kappa * k^2)
    growth_rates = -M * k ** 2 * (d2G + kappa * k ** 2)

    is_spinodal = d2G < 0

    if is_spinodal:
        # 临界波数: d2G + kappa * k_c^2 = 0
        k_c = np.sqrt(-d2G / kappa)
        # 最快增长波数
        k_m = np.sqrt(-d2G / (2.0 * kappa))
        # 最大增长率
        omega_m = M * d2G ** 2 / (4.0 * kappa)
    else:
        k_c = 0.0
        k_m = 0.0
        omega_m = 0.0

    return {
        'wavenumbers': k,
        'growth_rates': growth_rates,
        'k_critical': float(k_c),
        'k_max_growth': float(k_m),
        'omega_max': float(omega_m),
        'is_spinodal': bool(is_spinodal),
        'd2G_dc2': float(d2G),
    }


def free_energy_lyapunov_functional(c, T, h, kappa, phase):
    """
    计算 Cahn-Hilliard 自由能泛函 (Lyapunov 函数):

    V[c] = integral [G(c(x)) + (kappa/2) * |dc/dx|^2] dx

    该泛函满足 dV/dt <= 0 (CH 方程保证)。

    Parameters
    ----------
    c : np.ndarray
        浓度场
    T : float
        温度 (K)
    h : float
        空间步长
    kappa : float
        梯度能系数
    phase : str
        相名称

    Returns
    -------
    float
        总自由能 V (J/m)
    """
    from gibbs_energy_calphad import gibbs_substitutional
    from high_order_fd import compact_first_derivative

    N = len(c)
    # 局部化学自由能
    G_chem = gibbs_substitutional(c, T, phase)
    E_chem = np.sum(G_chem) * h

    # 梯度能
    dc_dx = compact_first_derivative(c, h)
    E_grad = 0.5 * kappa * np.sum(dc_dx ** 2) * h

    return float(E_chem + E_grad)


def spinodal_boundary_search(T, phase, x_range=None, n_points=100):
    """
    搜索 spinodal 边界 (d²G/dc² = 0 的点)。

    使用二分法在给定范围内搜索:
        d²G/dc²|_{x_spinodal} = 0

    Parameters
    ----------
    T : float
        温度 (K)
    phase : str
        相名称
    x_range : tuple
        (x_min, x_max) 搜索范围
    n_points : int
        初始扫描点数

    Returns
    -------
    list of float
        spinodal 成分点
    """
    from gibbs_energy_calphad import second_derivative_G

    if x_range is None:
        x_range = (1e-6, 0.25)

    x_scan = np.linspace(x_range[0], x_range[1], n_points)
    d2G = second_derivative_G(x_scan, T, phase)

    # 找符号变化
    spinodal_points = []
    for i in range(n_points - 1):
        if d2G[i] * d2G[i + 1] < 0:
            # 二分法精炼
            a, b = x_scan[i], x_scan[i + 1]
            for _ in range(50):
                mid = (a + b) / 2.0
                d2G_mid = second_derivative_G(mid, T, phase)
                if d2G_mid * second_derivative_G(a, T, phase) < 0:
                    b = mid
                else:
                    a = mid
            spinodal_points.append((a + b) / 2.0)

    return spinodal_points
