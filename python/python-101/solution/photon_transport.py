"""
photon_transport.py
===================
光子输运与局域化模拟

融合原项目:
  - 1091_snakes_and_ladders : 马尔可夫链转移矩阵 (光子跳跃概率)
  - 1070_shallow_water_1d   : 守恒律差分格式 (光子流密度演化)

本模块实现:
  1. 无序光子晶体中的光子扩散方程
  2. 基于马尔可夫链的光子跳跃模型
  3. 安德森局域化长度估算
  4. 辐射输运方程的数值求解
"""

import numpy as np
from physics_core import C_0


# =============================================================================
# 基于 1091_snakes_and_ladders 的马尔可夫链转移矩阵
# =============================================================================

def photon_hopping_matrix(n_sites, hopping_prob, disorder_strength):
    """
    构建光子跃迁的马尔可夫转移矩阵 —— 基于 snakes_matrix.m 思想
    
    将光子晶体离散为 N 个位点，光子以概率 p 跃迁到相邻位点。
    无序引入后，跃迁概率发生随机调制。
    
    转移矩阵 A 满足:
        P(t+1) = A · P(t)
    
    稳态分布对应 A 的左特征值 1 的特征向量。
    
    安德森局域化判据:
        若转移矩阵的所有特征值 (除 1 外) 的模 < 1，则系统趋向稳态
        局域化对应特征值谱出现间隙。
    
    Parameters
    ----------
    n_sites : int
        位点数
    hopping_prob : float
        最近邻跃迁概率 [0, 1]
    disorder_strength : float
        无序调制幅度 [0, 1]
    
    Returns
    -------
    A : ndarray, shape (n_sites, n_sites)
        转移矩阵
    eigenvalues : ndarray
        特征值谱
    localization_length : float
        估计局域化长度 (位点数)
    """
    if n_sites < 3:
        raise ValueError("位点数必须 >= 3")
    if not (0 <= hopping_prob <= 1):
        raise ValueError("跃迁概率必须在 [0, 1] 内")
    
    A = np.zeros((n_sites, n_sites))
    
    for i in range(n_sites):
        # 最近邻跃迁 (环形边界)
        neighbors = [(i - 1) % n_sites, (i + 1) % n_sites]
        
        for j in neighbors:
            # 无序调制跃迁概率
            delta = disorder_strength * (2.0 * np.random.rand() - 1.0)
            p_ij = hopping_prob * (1.0 + delta)
            p_ij = np.clip(p_ij, 0.0, 1.0)
            A[j, i] += p_ij / 2.0  # 列随机矩阵
        
        # 剩余概率为停留在原位 (模拟局域化)
        A[i, i] = max(0.0, 1.0 - np.sum(A[:, i]))
    
    # 归一化保证列和为 1
    for j in range(n_sites):
        col_sum = np.sum(A[:, j])
        if col_sum > 1e-12:
            A[:, j] /= col_sum
    
    # 计算特征值
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # 局域化长度估计: λ₂ 接近 1 对应长程输运
    # ξ ≈ -1 / ln(|λ₂|)
    if len(eigenvalues) > 1 and abs(eigenvalues[1]) < 1.0:
        if abs(eigenvalues[1]) > 1e-12:
            localization_length = -1.0 / np.log(abs(eigenvalues[1]))
        else:
            localization_length = 0.0
    else:
        localization_length = float('inf')
    
    return A, eigenvalues, localization_length


def photon_diffusion_markov(A, initial_distribution, n_steps):
    """
    用马尔可夫链模拟光子扩散
    
    P(t) = A^t · P(0)
    
    Parameters
    ----------
    A : ndarray
        转移矩阵
    initial_distribution : ndarray
        初始概率分布
    n_steps : int
        时间步数
    
    Returns
    -------
    distributions : ndarray, shape (n_steps+1, n_sites)
        每步的概率分布
    entropy : ndarray
        香农熵演化
    """
    n_sites = A.shape[0]
    P = np.asarray(initial_distribution, dtype=float)
    if len(P) != n_sites:
        raise ValueError("初始分布维度不匹配")
    
    P /= np.sum(P)  # 归一化
    
    distributions = np.zeros((n_steps + 1, n_sites))
    distributions[0, :] = P
    entropy = np.zeros(n_steps + 1)
    entropy[0] = -np.sum(P * np.log(P + 1e-18))
    
    for t in range(n_steps):
        P = A.dot(P)
        distributions[t + 1, :] = P
        entropy[t + 1] = -np.sum(P * np.log(P + 1e-18))
    
    return distributions, entropy


# =============================================================================
# 基于 1070_shallow_water_1d 思想的守恒律差分
# =============================================================================

def radiative_transfer_1d(I0, sigma_scat, sigma_abs, L, nz, n_angles=8):
    """
    一维辐射输运方程求解
    
    受 shallow_water_1d 中守恒律差分的启发，采用离散坐标法
    求解稳态辐射输运方程:
    
        μ ∂I/∂z + (σ_s + σ_a) I = σ_s/2 ∫ I(μ') dμ' + S
    
    其中:
        I(z, μ): 辐射强度
        μ = cos(θ): 方向余弦
        σ_s: 散射截面
        σ_a: 吸收截面
    
    Parameters
    ----------
    I0 : float
        入射辐射强度
    sigma_scat : float
        散射截面 [m⁻¹]
    sigma_abs : float
        吸收截面 [m⁻¹]
    L : float
        介质厚度 [m]
    nz : int
        空间网格数
    n_angles : int
        角度离散数
    
    Returns
    -------
    z : ndarray
        空间坐标 [m]
    I_forward : ndarray
        前向辐射强度
    I_backward : ndarray
        后向辐射强度
    transmittance : float
        透射率
    reflectance : float
        反射率
    """
    if L <= 0 or nz < 3:
        raise ValueError("参数超出允许范围")
    
    dz = L / (nz - 1)
    z = np.linspace(0, L, nz)
    
    sigma_total = sigma_scat + sigma_abs
    
    # 高斯-勒让德积分点 (简化用等间距)
    mu_pos = np.linspace(0.1, 0.9, n_angles // 2)  # 正向角度
    mu_neg = -mu_pos  # 负向角度
    
    w = np.ones(n_angles // 2) / (n_angles // 2)  # 权重
    
    I_plus = np.zeros((nz, n_angles // 2))   # μ > 0
    I_minus = np.zeros((nz, n_angles // 2))  # μ < 0
    
    # 边界条件
    I_plus[0, :] = I0
    I_minus[-1, :] = 0.0  # 无后向入射
    
    # 迭代求解 (类似于 Gauss-Seidel)
    max_iter = 1000
    tol = 1e-10
    
    for it in range(max_iter):
        I_plus_old = I_plus.copy()
        I_minus_old = I_minus.copy()
        
        # 正向扫描 (从左到右)
        for i in range(1, nz):
            for m in range(n_angles // 2):
                # 源项: 散射贡献
                source = 0.0
                for mp in range(n_angles // 2):
                    source += w[mp] * (I_plus[i - 1, mp] + I_minus[i - 1, mp])
                source *= sigma_scat / 2.0
                
                # 差分格式
                denominator = sigma_total + mu_pos[m] / dz
                if abs(denominator) < 1e-15:
                    denominator = 1e-15
                
                I_plus[i, m] = (mu_pos[m] / dz * I_plus[i - 1, m] + source) / denominator
        
        # 负向扫描 (从右到左)
        for i in range(nz - 2, -1, -1):
            for m in range(n_angles // 2):
                source = 0.0
                for mp in range(n_angles // 2):
                    source += w[mp] * (I_plus[i + 1, mp] + I_minus[i + 1, mp])
                source *= sigma_scat / 2.0
                
                denominator = sigma_total + abs(mu_neg[m]) / dz
                if abs(denominator) < 1e-15:
                    denominator = 1e-15
                
                I_minus[i, m] = (abs(mu_neg[m]) / dz * I_minus[i + 1, m] + source) / denominator
        
        # 收敛检查
        diff = np.max(np.abs(I_plus - I_plus_old)) + np.max(np.abs(I_minus - I_minus_old))
        if diff < tol:
            break
    
    # 计算透射率和反射率
    I_forward = np.sum(I_plus * w, axis=1)
    I_backward = np.sum(I_minus * w, axis=1)
    
    transmittance = I_forward[-1] / I0 if I0 > 1e-15 else 0.0
    reflectance = I_backward[0] / I0 if I0 > 1e-15 else 0.0
    
    transmittance = np.clip(transmittance, 0.0, 1.0)
    reflectance = np.clip(reflectance, 0.0, 1.0)
    
    return z, I_forward, I_backward, transmittance, reflectance


# =============================================================================
# 安德森局域化理论
# =============================================================================

def anderson_localization_length(wavelength, mean_free_path, disorder_strength):
    """
    估算三维无序介质中的安德森局域化长度
    
    Ioffe-Regel 判据:
        k·l ≤ 1 时发生强局域化
    
    局域化长度 (Vollhardt-Wölfle 理论):
        ξ_loc ≈ l · exp(π²/2 · (k·l)²)   (弱散射极限)
        ξ_loc ≈ l                         (强散射极限)
    
    其中 k = 2πn/λ 为波矢，l 为平均自由程。
    
    Parameters
    ----------
    wavelength : float
        真空波长 [m]
    mean_free_path : float
        输运平均自由程 [m]
    disorder_strength : float
        无序强度参数 [0, 1]
    
    Returns
    -------
    xi_loc : float
        局域化长度 [m]
    ioffe_regel : float
        Ioffe-Regel 参数 k·l
    is_localized : bool
        是否预测为局域化态
    """
    if wavelength <= 0 or mean_free_path <= 0:
        raise ValueError("波长和平均自由程必须为正")
    
    k = 2.0 * np.pi / wavelength
    kl = k * mean_free_path
    
    # Ioffe-Regel 判据
    is_localized = kl < 1.0
    
    if kl > 1.0:
        # 弱散射: Vollhardt-Wölfle
        xi_loc = mean_free_path * np.exp(np.pi ** 2 / 2.0 * kl ** 2)
    else:
        # 强散射: 局域化长度 ~ 平均自由程
        xi_loc = mean_free_path * (1.0 + 0.5 * (1.0 - kl))
    
    # 无序修正
    xi_loc *= (1.0 - 0.3 * disorder_strength)
    
    return xi_loc, kl, is_localized


def photon_mean_free_path(eps_r, wavelength, correlation_length):
    """
    从介电常数分布估算光子平均自由程
    
    采用 Rayleigh 散射近似 (使用相对涨落):
        1/l = (π²/λ⁴) · (Δε/ε̄)² · L_c³ · ε̄²
    
    其中 Δε/ε̄ 为相对介电常数涨落。
    
    Parameters
    ----------
    eps_r : ndarray
        相对介电常数分布
    wavelength : float
        真空波长 [m]
    correlation_length : float
        介电涨落关联长度 [m]
    
    Returns
    -------
    l_mfp : float
        平均自由程 [m]
    delta_eps_rms : float
        相对介电常数 RMS 涨落
    """
    if wavelength <= 0 or correlation_length <= 0:
        raise ValueError("参数必须为正")
    
    eps_mean = np.mean(eps_r)
    if eps_mean < 1e-12:
        eps_mean = 1.0
    
    delta_eps = (eps_r - eps_mean) / eps_mean
    delta_eps_rms = np.sqrt(np.mean(delta_eps ** 2))
    
    if delta_eps_rms < 1e-12:
        return float('inf'), 0.0
    
    # Rayleigh 散射截面 (使用相对涨落)
    scattering_cross_section = (np.pi ** 2 / wavelength ** 4) * (delta_eps_rms ** 2) * (correlation_length ** 3) * (eps_mean ** 2)
    
    # 散射体数密度
    n_scatterers = 1.0 / (correlation_length ** 3)
    
    l_mfp_rayleigh = 1.0 / max(n_scatterers * scattering_cross_section, 1e-30)
    
    # 对于强散射介质，Rayleigh 公式可能给出非物理的极小值
    # 采用启发式修正: l_mfp ≈ correlation_length / (delta_eps_rms)^2
    l_mfp_heuristic = correlation_length / max(delta_eps_rms ** 2, 1e-12)
    
    # 取两者中更合理的一个
    l_mfp = max(min(l_mfp_rayleigh, l_mfp_heuristic), wavelength * 1e-3)
    
    # 物理边界保护
    l_mfp = min(l_mfp, wavelength * 1e4)
    
    return l_mfp, delta_eps_rms


def diffusion_constant_photonic(l_mfp, v_group):
    """
    光子扩散常数
    
    公式:
        D = (1/3) · v_g · l_mfp
    
    Parameters
    ----------
    l_mfp : float
        平均自由程 [m]
    v_group : float
        群速度 [m/s]
    
    Returns
    -------
    float
        扩散常数 [m²/s]
    """
    if l_mfp < 0 or v_group < 0:
        raise ValueError("参数必须非负")
    return (1.0 / 3.0) * v_group * l_mfp


def scaling_theory_beta_function(g, d=3):
    """
    标度理论中的 β 函数 (Abrahams et al. 1979)
    
    无量纲电导 g 的标度方程:
        d(ln g)/d(ln L) = β(g)
    
    渐近行为:
        g → ∞:  β(g) = d - 2   (欧几里得维度)
        g → 0:  β(g) = ln g    (局域化极限)
    
    插值公式 (Vollhardt-Wölfle):
        β(g) = (d - 2) + (2 - d) / (1 + g²)
    
    Parameters
    ----------
    g : float
        无量纲 Thouless 电导
    d : int
        空间维度
    
    Returns
    -------
    float
        β(g) 值
    """
    if g <= 0:
        return -float('inf')
    
    # 插值公式
    beta = (d - 2.0) + (2.0 - d) / (1.0 + g ** 2)
    return beta
