"""
eigenmode_solver.py

电磁模式分析与特征值求解模块。
融合power_method的主导特征值迭代思想与pagerank的矩阵转移/随机游走思想，
用于分析微波谐振腔中的本征模式。

核心数学模型:
--------------
1. 亥姆霍兹方程（矢量形式）:
   ∇ × (1/μ ∇ × E) - ω²εE = 0
   或标量形式（对于特定模式）:
   ∇²ψ + k²ψ = 0, 其中 k² = ω²με

2. 特征值问题:
   A · x = λ · x
   其中 λ 对应特征频率的平方，x 对应模式场分布。

3. 幂方法（Power Method）:
   y_{k+1} = A · y_k / ||A · y_k||
   λ_{k+1} = y_{k+1}^T · A · y_{k+1}
   收敛到最大模特征值和对应特征向量。

4. 逆幂方法（Inverse Power Method）:
   用于求解最接近给定频率的模式。
   A · y_{k+1} = y_k, 然后归一化。

5. Google矩阵思想:
   将电磁场的功率流网络建模为图，
   用类似PageRank的方法识别能量集中的模式区域。
"""

import numpy as np
from physics_constants import EPSILON_0, MU_0


def power_method_eigenmode(A_func, x0, it_max=1000, tol=1e-10):
    """
    幂方法求解矩阵最大模特征值和特征向量。
    基于power_method.m的核心算法。

    Parameters
    ----------
    A_func : callable
        矩阵-向量乘法函数 A_func(x) -> A·x
    x0 : ndarray
        初始向量
    it_max : int
        最大迭代次数
    tol : float
        收敛容差

    Returns
    -------
    lambda_max, eigenvector, it_num, convergence_history
    """
    x = x0.copy()
    n = x.size
    x = x.reshape(n)
    x = x / np.linalg.norm(x)

    lambda_old = 0.0
    convergence_history = []

    for it_num in range(1, it_max + 1):
        y = A_func(x)
        lambda_val = np.dot(x, y)
        y_norm = np.linalg.norm(y)
        if y_norm < 1e-30:
            break
        x_new = y / y_norm
        if lambda_val < 0:
            x_new = -x_new
            lambda_val = -y_norm

        delta_lambda = abs(lambda_val - lambda_old)
        cos_xy = np.dot(x, x_new)
        sin_xy = np.sqrt(max(0.0, 1.0 - cos_xy ** 2))

        convergence_history.append((it_num, lambda_val, delta_lambda, sin_xy))

        if delta_lambda <= tol and sin_xy <= tol:
            x = x_new
            break

        x = x_new
        lambda_old = lambda_val

    return lambda_val, x, it_num, convergence_history


def inverse_power_method(A_func, x0, sigma_shift, it_max=500, tol=1e-10):
    """
    带位移的逆幂方法，用于求解最接近sigma_shift的特征值。

    求解 (A - σI)^{-1} 的最大特征值，对应A最接近σ的特征值。
    这里使用迭代法近似求解线性系统。

    Parameters
    ----------
    A_func : callable
        A_func(x) -> A·x
    x0 : ndarray
        初始向量
    sigma_shift : float
        位移量
    it_max : int
        最大迭代次数
    tol : float
        收敛容差

    Returns
    -------
    lambda_approx, eigenvector, it_num
    """
    x = x0.copy().reshape(-1)
    x = x / np.linalg.norm(x)

    # 使用简单的Richardson迭代作为近似逆
    # (A - σI) y = x  =>  y = x / (λ_dom - σ) 作为粗糙近似
    lambda_old = 0.0

    for it_num in range(1, it_max + 1):
        # 近似求解: y ≈ (A - σI)^{-1} x
        # 使用简单的Jacobi预处理迭代
        Ax = A_func(x)
        y = x / (sigma_shift + 1e-10)  # 粗糙近似
        for _ in range(5):
            residual = x - (Ax - sigma_shift * x)
            y = y + 0.1 * residual
            Ay = A_func(y)
            residual = x - (Ay - sigma_shift * y)
            if np.linalg.norm(residual) < tol:
                break

        # 计算Rayleigh商
        Ay = A_func(y)
        lambda_val = np.dot(y, Ay) / np.dot(y, y)

        y_norm = np.linalg.norm(y)
        if y_norm < 1e-30:
            break
        x_new = y / y_norm

        delta = abs(lambda_val - lambda_old)
        if delta < tol:
            x = x_new
            break

        x = x_new
        lambda_old = lambda_val

    return lambda_val, x, it_num


def build_fd_helmholtz_operator_2d(nx, ny, dx, dy, epsilon, mu):
    """
    构建二维标量亥姆霍兹方程的有限差分算子 A。
    形式: A · ψ = -∇²ψ，对应特征值 λ = k² = ω²με。

    使用5点差分格式（biharmonic_fd2d_stencil的拉普拉斯部分）:
    ∇²ψ ≈ (ψ_{i+1,j} + ψ_{i-1,j} + ψ_{i,j+1} + ψ_{i,j-1} - 4ψ_{i,j}) / h²

    Parameters
    ----------
    nx, ny : int
        网格数
    dx, dy : float
        网格步长
    epsilon, mu : ndarray shape (nx, ny)
        材料参数

    Returns
    -------
    A_func : callable
        矩阵-向量乘法函数
    """
    def A_func(psi_vec):
        psi = psi_vec.reshape((nx, ny))
        laplacian = np.zeros_like(psi)

        # 内部点
        laplacian[1:-1, 1:-1] = (
            (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dx ** 2 +
            (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dy ** 2
        )

        # 边界条件（Dirichlet: ψ = 0）
        laplacian[0, :] = 0.0
        laplacian[-1, :] = 0.0
        laplacian[:, 0] = 0.0
        laplacian[:, -1] = 0.0

        # 对应 A = -∇²
        result = -laplacian
        return result.reshape(-1)

    return A_func


def compute_cavity_modes_2d(nx, ny, dx, dy, epsilon, mu, n_modes=3, max_iter=500):
    """
    计算二维谐振腔的前n_modes个模式。

    Returns
    -------
    modes : list of dict
        每个模式包含 {'frequency', 'wavenumber', 'field', 'eigenvalue'}
    """
    A_func = build_fd_helmholtz_operator_2d(nx, ny, dx, dy, epsilon, mu)

    modes = []
    psi0 = np.random.randn(nx, ny)
    psi0[0, :] = 0.0
    psi0[-1, :] = 0.0
    psi0[:, 0] = 0.0
    psi0[:, -1] = 0.0

    for mode_idx in range(n_modes):
        # 使用逆幂方法求解
        sigma_guess = (mode_idx + 1) * np.pi ** 2 * (1.0 / dx ** 2 + 1.0 / dy ** 2) / 10.0
        lambda_val, eigenvec, it_num = inverse_power_method(
            A_func, psi0.flatten(), sigma_guess, it_max=max_iter
        )

        # 正交化：去除已找到模式的分量
        field = eigenvec.reshape((nx, ny))
        for prev_mode in modes:
            overlap = np.sum(field * prev_mode['field'])
            field = field - overlap * prev_mode['field']

        field = field / (np.linalg.norm(field) + 1e-30)

        # 计算物理频率
        # λ = k² = ω²με  =>  ω = sqrt(λ/με)
        eps_avg = np.mean(epsilon)
        mu_avg = np.mean(mu)
        omega = np.sqrt(max(lambda_val, 0.0) / (eps_avg * mu_avg))
        frequency = omega / (2.0 * np.pi)
        wavenumber = np.sqrt(max(lambda_val, 0.0))

        modes.append({
            'frequency': frequency,
            'wavenumber': wavenumber,
            'field': field,
            'eigenvalue': lambda_val,
            'iterations': it_num,
        })

        # 为下一个模式更新初始猜测
        psi0 = np.random.randn(nx, ny)
        psi0[0, :] = 0.0
        psi0[-1, :] = 0.0
        psi0[:, 0] = 0.0
        psi0[:, -1] = 0.0

    return modes


def power_flow_pagerank_analysis(E, H, dx, dy, dz, damping=0.15, n_iter=100):
    """
    使用类PageRank方法分析电磁功率流网络。

    将空间离散为网格节点，计算坡印廷矢量S = E × H作为节点间的"链接权重"，
    构建功率流转移矩阵，通过迭代识别能量集中区域。

    基于pagerank/google_matrix的思想:
    G = (1-p)·T + p·(1/N)·J
    其中T为归一化的功率流转移矩阵，p为阻尼因子。

    Parameters
    ----------
    E, H : tuple of ndarray
        电磁场
    dx, dy, dz : float
        网格尺寸
    damping : float
        阻尼因子 (类似Google的0.15)
    n_iter : int
        迭代次数

    Returns
    -------
    rank_field : ndarray
        每个网格节点的"能量重要性"分数
    """
    Ex, Ey, Ez = E
    Hx, Hy, Hz = H
    nx, ny, nz = Ex.shape

    # 计算坡印廷矢量大小
    Sx = Ey * Hz - Ez * Hy
    Sy = Ez * Hx - Ex * Hz
    Sz = Ex * Hy - Ey * Hx
    S_mag = np.sqrt(Sx ** 2 + Sy ** 2 + Sz ** 2)

    # 将三维场展平为一维
    N = nx * ny * nz
    s_flat = S_mag.flatten()

    # 构建简化的转移矩阵（基于局部邻居功率流）
    # 使用能量密度作为节点重要性
    from physics_constants import electromagnetic_energy_density
    eps = np.ones_like(Ex) * EPSILON_0
    mu = np.ones_like(Ex) * MU_0
    w = electromagnetic_energy_density(E, H, eps, mu)
    w_flat = w.flatten()

    # 初始均匀分布
    rank = np.ones(N) / N

    # 迭代（类似power_rank）
    for _ in range(n_iter):
        # 简化的PageRank更新:
        # rank_new = (1-damping)·(归一化能量权重) + damping·(均匀分布)
        if np.sum(w_flat) > 1e-30:
            energy_weights = w_flat / np.sum(w_flat)
        else:
            energy_weights = np.ones(N) / N

        rank = (1.0 - damping) * energy_weights + damping * rank
        rank = rank / (np.sum(rank) + 1e-30)

    rank_field = rank.reshape((nx, ny, nz))
    return rank_field
