"""
population_balance.py
=====================
基于 879_poisson_simulation 改造的群体平衡方程（PBE）模块。

在气泡柱反应器中，气泡尺寸分布（BSD）随时间和空间演变，直接影响
气含率、相间面积与传质速率。本模块实现：
1. 气泡成核事件的 Poisson 过程模拟
2. 群体平衡方程的矩方法（QMOM）求解
3. 破裂与聚并核函数
4. 基于稀疏网格的高维矩积分

核心公式
--------
1. 群体平衡方程（PBE）：
       ∂f(V; x,t)/∂t + ∇·[u_b(V) f] = B_B(V) - D_B(V) + B_C(V) - D_C(V)
   其中 f(V) 为气泡体积概率密度函数 [1/m³/m³]。

2. 破裂项：
       B_B(V) = ∫_V^∞ ν(V') g(V') f(V') dV'
       D_B(V) = g(V) f(V)
   ν(V') 为母气泡破裂产生的子气泡数，
   g(V)  为破裂频率 [1/s]。

3. 聚并项：
       B_C(V) = 1/2 ∫_0^V Q(V-V', V') f(V-V') f(V') dV'
       D_C(V) = f(V) ∫_0^∞ Q(V, V') f(V') dV'
   Q(V, V') = ω(V, V') h(V, V')
   ω 为碰撞频率，h 为碰撞效率。

4. 破裂频率（Lehr-Mewes 模型）：
       g(V) = C_B · √(σ / (ρ_l d_eq³))
   其中 d_eq = (6V/π)^{1/3}。

5. 聚并效率（Prince-Blanch 模型）：
       h(V_i, V_j) = exp(-t_contact / t_drainage)
       t_contact = (r_i + r_j)^{2/3} / ε^{1/3}
       t_drainage = √(ρ_l r_eq³ / (16 σ)) ln(h_0 / h_f)

6. 矩方法（QMOM）：
       m_k = ∫_0^∞ V^k f(V) dV
       ∂m_k/∂t + ∇·(u_b m_k) = S̄_k
   其中源项 S̄_k 通过 Gaussian 积分节点与权重计算：
       S̄_k ≈ Σ_{i=1}^{N} w_i [B_B(V_i) - D_B(V_i) + B_C(V_i) - D_C(V_i)] V_i^k

7. 气泡成核（Poisson 过程）：
       N(t) ~ Poisson(λ t)
       等待时间 W_i ~ Exp(λ)
       其中 λ 为单位时间单位体积的成核率。
"""

import numpy as np
from spectral_quadrature import legendre_nodes_weights, sparse_grid_gauss_legendre


# ---------------------------------------------------------------------------
# Poisson nucleation simulation (from 879_poisson_simulation)
# ---------------------------------------------------------------------------

def poisson_nucleation_events(lambda_rate, t_end, event_num=None, seed=42):
    """
    模拟气泡成核的 Poisson 过程。

    Parameters
    ----------
    lambda_rate : float
        成核率 [events/s]。
    t_end : float
        总时间 [s]。
    event_num : int or None
        若指定，则生成固定数目事件；否则按 Poisson 分布生成。
    seed : int
        随机种子。

    Returns
    -------
    t : ndarray
        事件发生的绝对时间 [s]。
    w : ndarray
        等待时间 [s]。
    n_total : int
        总事件数。
    """
    rng = np.random.default_rng(seed)
    if event_num is None:
        n_total = rng.poisson(lam=lambda_rate * t_end)
    else:
        n_total = event_num

    if n_total <= 0:
        return np.array([0.0]), np.array([0.0]), 0

    w = np.zeros(n_total + 1)
    w[1:] = rng.exponential(scale=1.0 / lambda_rate, size=n_total)
    t = np.cumsum(w)

    # 只保留在 t_end 内的事件
    mask = t <= t_end
    t = t[mask]
    w = w[mask]
    n_total = t.size - 1 if t.size > 0 else 0
    return t, w, n_total


# ---------------------------------------------------------------------------
# Breakage and coalescence kernels
# ---------------------------------------------------------------------------

def breakage_frequency_lehr(V, C_B=0.5, sigma=0.072, rho_l=800.0):
    """
    Lehr-Mewes 破裂频率 [1/s]。

    Parameters
    ----------
    V : float or ndarray
        气泡体积 [m³]。
    C_B : float
        模型常数。
    sigma : float
        表面张力 [N/m]。
    rho_l : float
        液相密度 [kg/m³]。
    """
    V = np.asarray(V, dtype=float)
    V = np.clip(V, 1e-15, None)
    d_eq = (6.0 * V / np.pi) ** (1.0 / 3.0)
    return C_B * np.sqrt(sigma / (rho_l * d_eq**3))


def daughter_distribution_uniform(V_parent, V_daughter):
    """
    二元均匀破裂：母气泡 V_parent 破裂为两个子气泡，
    子气泡体积分布为在 [0, V_parent] 上的均匀分布。
    这里返回 ν(V_parent) 的期望子气泡数 = 2。
    """
    return 2.0


def coalescence_kernel_prince_blanch(V_i, V_j, epsilon=0.1, sigma=0.072,
                                      rho_l=800.0, h0_hf_ratio=10.0):
    """
    Prince-Blanch 聚并核函数 [m³/s]。

    Parameters
    ----------
    V_i, V_j : float or ndarray
        气泡体积 [m³]。
    epsilon : float
        湍流耗散率 [m²/s³]。
    sigma : float
        表面张力 [N/m]。
    rho_l : float
        液相密度 [kg/m³]。
    h0_hf_ratio : float
        初始液膜厚度与临界厚度之比。
    """
    V_i = np.asarray(V_i, dtype=float)
    V_j = np.asarray(V_j, dtype=float)
    V_i = np.clip(V_i, 1e-15, None)
    V_j = np.clip(V_j, 1e-15, None)

    r_i = (3.0 * V_i / (4.0 * np.pi)) ** (1.0 / 3.0)
    r_j = (3.0 * V_j / (4.0 * np.pi)) ** (1.0 / 3.0)
    r_eq = 2.0 * r_i * r_j / (r_i + r_j)

    # 碰撞频率（湍流驱动）
    omega = 1.43 * epsilon ** (1.0 / 3.0) * (r_i + r_j) ** 2.0

    # 接触时间
    t_contact = (r_i + r_j) ** (2.0 / 3.0) / (epsilon ** (1.0 / 3.0) + 1e-12)

    # 液膜排液时间
    t_drainage = np.sqrt(rho_l * r_eq ** 3.0 / (16.0 * sigma + 1e-12)) * np.log(h0_hf_ratio)

    # 聚并效率
    h = np.exp(-t_contact / (t_drainage + 1e-12))
    h = np.clip(h, 0.0, 1.0)

    return omega * h


# ---------------------------------------------------------------------------
# Quadrature Method of Moments (QMOM)
# ---------------------------------------------------------------------------

def moment_source_qmom(moments, xi, wi, rho_l=800.0, sigma=0.072, epsilon=0.1):
    """
    计算 PBE 矩方程的源项 S̄_k（k=0,1,2,3）。

    Parameters
    ----------
    moments : ndarray, shape (4,)
        当前时刻的矩 m0, m1, m2, m3。
    xi : ndarray, shape (N,)
        体积节点的横坐标（通过 Wheeler 算法或已知分布得到）。
    wi : ndarray, shape (N,)
        权重。
    rho_l, sigma, epsilon : float
        物性参数。

    Returns
    -------
    S_bar : ndarray, shape (4,)
        各阶矩的源项。
    """
    # TODO: 实现 PBE 矩方程的源项 S̄_k 计算（k=0,1,2,3）
    # 包括：
    # 1. 破裂源项（Lehr-Mewes 破裂频率 + 二元均匀破裂假设）
    # 2. 聚并源项（Prince-Blanch 聚并核 + 体积守恒）
    # 注意：xi 为 Wheeler 算法得到的体积节点，wi 为对应权重
    raise NotImplementedError("Hole 3: 请实现 moment_source_qmom 的矩源项计算")


def wheeler_algorithm(moments, n_nodes=2):
    """
    Wheeler 算法：由低阶矩构造 Gaussian 积分节点与权重。
    对 n_nodes=2 采用解析公式，数值更稳定；
    对 n_nodes>2 采用 Hankel 矩阵特征值分解（简化实现）。

    Parameters
    ----------
    moments : ndarray, shape (2*n_nodes,)
        矩序列 m_0, m_1, ..., m_{2n-1}。
    n_nodes : int
        节点数。

    Returns
    -------
    xi : ndarray
        节点（体积坐标）。
    wi : ndarray
        权重。
    """
    moments = np.asarray(moments, dtype=float)
    n = n_nodes
    if len(moments) < 2 * n:
        raise ValueError("Need at least 2*n moments")

    m0 = moments[0]
    if m0 <= 0:
        m0 = 1e-12
    mu = moments / m0  # 归一化矩

    if n == 2:
        # 解析求解 2 节点 Gaussian 积分
        # 构造正交多项式 P(x) = x^2 - a*x + b
        # 系数由线性方程组确定：
        #   m0*b - m1*a + m2 = 0
        #   m1*b - m2*a + m3 = 0
        det = -mu[0] * mu[2] + mu[1] ** 2
        if abs(det) < 1e-30:
            # 退化情形（单分散或接近单分散）
            x0 = max(mu[1], 1e-15)
            xi = np.array([x0 * 0.9, x0 * 1.1])
            wi = np.array([m0 * 0.5, m0 * 0.5])
            return xi, wi

        b = (mu[2] ** 2 - mu[1] * mu[3]) / det
        a = (mu[0] * mu[3] - mu[1] * mu[2]) / det

        disc = a ** 2 - 4.0 * b
        if disc < 0.0:
            disc = 0.0
        sqrt_disc = np.sqrt(disc)
        x1 = (a - sqrt_disc) / 2.0
        x2 = (a + sqrt_disc) / 2.0

        # 非负约束与排序
        x1 = max(x1, 1e-15)
        x2 = max(x2, 1e-15)

        if abs(x2 - x1) < 1e-15:
            w1 = w2 = m0 * 0.5
        else:
            w1 = (m0 * x2 - moments[1]) / (x2 - x1)
            w2 = (moments[1] - m0 * x1) / (x2 - x1)

        w1 = max(w1, 0.0)
        w2 = max(w2, 0.0)

        # 若权重异常（如因节点极不对称），退化为等权重
        if w1 + w2 < 1e-15 or x1 > 1e3 * x2 or x2 > 1e3 * x1:
            x_avg = max(mu[1], 1e-15)
            xi = np.array([x_avg * 0.8, x_avg * 1.2])
            wi = np.array([m0 * 0.5, m0 * 0.5])
            return xi, wi

        return np.array([x1, x2]), np.array([w1, w2])

    else:
        # 通用 Hankel 矩阵方法（n>2）
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                idx = i + j
                if idx < len(mu):
                    H[i, j] = mu[idx]
        H = H + 1e-14 * np.eye(n)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            xi = np.clip(eigenvalues, 1e-15, None)
            wi = m0 * (eigenvectors[0, :] ** 2)
            wi = np.clip(wi, 0.0, None)
        except np.linalg.LinAlgError:
            xi = np.linspace(1e-6, 1.0, n)
            wi = np.ones(n) * m0 / n
        return xi, wi


def qmom_integrate_pbe(m0_init, t_span, dt, n_nodes=2, **kwargs):
    """
    用 QMOM 求解 PBE 的矩方程。

    Parameters
    ----------
    m0_init : ndarray, shape (4,)
        初始矩 [m0, m1, m2, m3]。
    t_span : tuple
        (t0, tf)。
    dt : float
        时间步长。
    n_nodes : int
        Wheeler 算法的节点数。

    Returns
    -------
    t_array : ndarray
    moments_history : ndarray, shape (nt, 4)
    """
    t0, tf = t_span
    t_array = np.arange(t0, tf + dt, dt)
    nt = len(t_array)
    moments_hist = np.zeros((nt, 4))
    moments = np.asarray(m0_init, dtype=float).copy()
    moments_hist[0] = moments

    for it in range(1, nt):
        try:
            xi, wi = wheeler_algorithm(moments, n_nodes=n_nodes)
            S = moment_source_qmom(moments, xi, wi, **kwargs)
        except Exception:
            # 若 Wheeler 失败，保持上一时刻
            S = np.zeros(4)

        # 显式 Euler 时间推进（可替换为 RK4）
        moments = moments + dt * S
        moments = np.clip(moments, 0.0, None)  # 矩非负
        moments_hist[it] = moments

    return t_array, moments_hist
