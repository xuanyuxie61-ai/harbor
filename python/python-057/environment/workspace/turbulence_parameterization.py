"""
turbulence_parameterization.py
海洋湍流混合参数化

融合项目:
- 1416_wishart_matrix: Wishart分布 → 雷诺应力张量采样
- 194_cobweb_plot: 不动点迭代 → 混合效率不动点求解
- 776_monomial_symmetrize: 单项式对称化 → 波数空间对称性处理

核心科学:
海洋内波破碎产生的湍流混合需要参数化描述。

1. 雷诺应力张量:
    τ_{ij} = -ρ₀ <u'_i u'_j>
    
    使用Wishart分布采样:
    W ~ W_m(Σ, df)  其中 Σ 为背景协方差矩阵
    
    Bartlett分解:
    W = T' · T  其中 T 为上三角随机矩阵

2. 混合效率:
    Γ = ε_b / ε
    
    其中 ε_b 为浮力通量耗散，ε 为总湍流耗散。
    
    混合效率的不动点方程:
    Γ = f(Γ) = Γ_max / (1 + α · Ri · Γ)

3. 波数空间对称化:
    利用单项式对称化保证能量谱在波数空间的置换对称性。
"""

import numpy as np


def bartlett_sample(m, df):
    """
    Bartlett分解采样上三角随机矩阵
    
    C(i,i) = √(χ²(df - i + 1))
    C(i,j) = N(0,1)  (i < j)
    
    参数:
        m: 矩阵维度
        df: 自由度
    
    返回:
        C: 上三角矩阵
    """
    C = np.zeros((m, m))
    
    for i in range(m):
        # 对角线: chi-square
        df_chi = max(df - i, 1)
        C[i, i] = np.sqrt(np.random.chisquare(df_chi))
        
        # 上三角: 标准正态
        for j in range(i + 1, m):
            C[i, j] = np.random.normal(0.0, 1.0)
    
    return C


def wishart_sample(m, df, sigma):
    """
    Wishart分布采样
    
    W = R' · AU · R  其中 R = chol(Σ), AU ~ W_m(I, df)
    
    参数:
        m: 矩阵维度
        df: 自由度
        sigma: 协方差矩阵 (m x m)
    
    返回:
        W: Wishart随机矩阵
    """
    # Cholesky分解
    try:
        R = np.linalg.cholesky(sigma).T  # 上三角
    except np.linalg.LinAlgError:
        # 若不正定，添加微小扰动
        sigma = sigma + 1.0e-6 * np.eye(m)
        R = np.linalg.cholesky(sigma).T
    
    # 单位Wishart采样
    C = bartlett_sample(m, df)
    AU = C.T @ C
    
    # 变换
    W = R.T @ AU @ R
    
    # 确保对称正定
    W = 0.5 * (W + W.T)
    eigvals = np.linalg.eigvalsh(W)
    if np.min(eigvals) < 1.0e-10:
        W = W + (1.0e-10 - np.min(eigvals)) * np.eye(m)
    
    return W


def sample_reynolds_stress_tensor(shear_magnitude=0.01,
                                   buoyancy_flux=1.0e-7,
                                   m=3, df=10):
    """
    采样雷诺应力张量
    
    背景协方差矩阵:
        Σ = diag(τ_11, τ_22, τ_33)
    
    参数:
        shear_magnitude: 剪切强度 [1/s]
        buoyancy_flux: 浮力通量 [m²/s³]
        m: 张量维度
        df: 自由度
    
    返回:
        tau: 雷诺应力张量 [Pa]
    """
    # 背景协方差 (基于剪切和浮力)
    tau_11 = shear_magnitude**2
    tau_22 = shear_magnitude**2
    tau_33 = buoyancy_flux
    
    sigma = np.diag([tau_11, tau_22, tau_33])
    
    W = wishart_sample(m, df, sigma)
    
    # 转换为雷诺应力 (取负值)
    rho0 = 1025.0
    tau = -rho0 * W / df
    
    return tau


def mixing_efficiency_fixed_point(Ri, gamma_max=0.2, alpha=5.0,
                                   max_iter=100, tol=1.0e-8):
    """
    混合效率不动点迭代求解
    
    不动点方程 (基于cobweb_plot思想):
        Γ_{n+1} = f(Γ_n) = Γ_max / (1 + α · Ri · Γ_n)
    
    物理意义:
        当Ri较小时，混合效率高;
        当Ri增大时，混合效率降低。
    
    参数:
        Ri: Richardson数
        gamma_max: 最大混合效率
        alpha: 衰减系数
        max_iter: 最大迭代次数
        tol: 收敛容差
    
    返回:
        gamma: 收敛的混合效率
        history: 迭代历史
        converged: 是否收敛
    """
    Ri = np.asarray(Ri)
    
    # 初始猜测
    gamma = gamma_max * 0.5
    history = [gamma]
    
    converged = False
    
    for _ in range(max_iter):
        gamma_new = gamma_max / (1.0 + alpha * Ri * gamma)
        
        # 边界处理
        gamma_new = np.clip(gamma_new, 0.0, gamma_max)
        
        history.append(gamma_new)
        
        if np.abs(gamma_new - gamma) < tol:
            converged = True
            break
        
        gamma = gamma_new
    
    return gamma, np.array(history), converged


def cobweb_iteration_analysis(Ri_values, gamma_max=0.2, alpha=5.0):
    """
    混合效率的蛛网图分析
    
    分析不动点迭代在不同Ri值下的收敛行为。
    
    参数:
        Ri_values: Richardson数数组
        gamma_max: 最大混合效率
        alpha: 衰减系数
    
    返回:
        results: 字典，包含各Ri下的不动点结果
    """
    results = {}
    
    for Ri in Ri_values:
        gamma, history, converged = mixing_efficiency_fixed_point(
            Ri, gamma_max, alpha
        )
        results[Ri] = {
            'gamma': gamma,
            'history': history,
            'converged': converged,
            'n_iter': len(history)
        }
    
    return results


def monomial_symmetrize_2d(coefficients, n_kx=4, n_kz=4):
    """
    2D波数空间单项式对称化
    
    将波数空间的系数按照置换群对称化:
        c_sym(kx, kz) = mean_{π ∈ S_2} c(π(kx, kz))
    
    参数:
        coefficients: (n_kx, n_kz) 系数矩阵
        n_kx, n_kz: 波数维度
    
    返回:
        sym_coeffs: 对称化系数
    """
    coeffs = np.asarray(coefficients)
    sym_coeffs = coeffs.copy()
    
    # 对于2D，S_2群只有2个元素: 恒等和交换
    # 对称化: c_sym = (c(kx,kz) + c(kz,kx)) / 2
    
    min_dim = min(coeffs.shape[0], coeffs.shape[1])
    
    for i in range(min_dim):
        for j in range(i + 1, min_dim):
            avg = 0.5 * (coeffs[i, j] + coeffs[j, i])
            sym_coeffs[i, j] = avg
            sym_coeffs[j, i] = avg
    
    return sym_coeffs


def symmetrize_wave_spectrum(E_kx_kz):
    """
    对称化内波能量谱
    
    保证能量谱满足:
        E(kx, kz) = E(-kx, kz) = E(kx, -kz) = E(-kx, -kz)
    
    参数:
        E_kx_kz: 波数空间能量谱
    
    返回:
        E_sym: 对称化能量谱
    """
    E = np.asarray(E_kx_kz)
    E_sym = E.copy()
    
    nx, nz = E.shape
    
    # 对称化操作
    for i in range(nx):
        for j in range(nz):
            # 四个对称点
            i_mirror = (nx - 1 - i) % nx
            j_mirror = (nz - 1 - j) % nz
            
            sym_val = 0.25 * (E[i, j] + E[i_mirror, j] +
                              E[i, j_mirror] + E[i_mirror, j_mirror])
            
            E_sym[i, j] = sym_val
            E_sym[i_mirror, j] = sym_val
            E_sym[i, j_mirror] = sym_val
            E_sym[i_mirror, j_mirror] = sym_val
    
    return E_sym
