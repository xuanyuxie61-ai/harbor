"""
maxwell_eigensolver.py
======================
麦克斯韦方程组本征值求解器

融合原项目:
  - 991_r8pp          : 对称正定矩阵的 packed Cholesky 分解
  - 451_gauss_seidel  : 迭代线性方程组求解
  - 1154_st_to_ccs    : 稀疏矩阵 ST→CCS 格式转换
  - 1085_sine_transform: 离散正弦变换谱方法

本模块实现二维 TE/TM 模式麦克斯韦本征问题的多种数值解法，
包括平面波展开 (PWE)、有限差分时域 (FDTD) 预处理、以及
基于离散正弦变换的谱方法。
"""

import numpy as np
from physics_core import C_0, helmholtz_operator_2d_te


# =============================================================================
# 基于 991_r8pp 的对称正定矩阵 Cholesky 分解 (packed 存储)
# =============================================================================

def r8pp_fa(n, a):
    """
    对称正定矩阵的 packed Cholesky 分解 —— 基于 r8pp_fa.m
    
    存储格式: 上三角按列压缩存储
        A_packed = [A11, A12, A22, A13, A23, A33, A14, ...]
    
    分解:
        A = R' · R
    
    其中 R 为上三角矩阵。
    
    Parameters
    ----------
    n : int
        矩阵阶数
    a : ndarray, shape (n*(n+1)//2,)
        packed 存储的对称正定矩阵
    
    Returns
    -------
    r : ndarray
        packed 存储的 R 因子
    info : int
        0=成功, K=第 K 阶主子式非正定
    """
    if n < 1:
        raise ValueError("矩阵阶数必须 >= 1")
    expected_len = n * (n + 1) // 2
    if len(a) != expected_len:
        raise ValueError(f"packed 数组长度应为 {expected_len}，实际为 {len(a)}")
    
    r = a.copy()
    info = 0
    
    for j in range(n):
        s = 0.0
        for k in range(j):
            idx_kj = k + j * (j + 1) // 2
            t = r[idx_kj]
            for i in range(k):
                idx_ik = i + k * (k + 1) // 2
                idx_ij = i + j * (j + 1) // 2
                t -= r[idx_ik] * r[idx_ij]
            idx_kk = k + k * (k + 1) // 2
            if abs(r[idx_kk]) < 1e-15:
                info = k + 1
                return r, info
            t /= r[idx_kk]
            r[idx_kj] = t
            s += t * t
        
        idx_jj = j + j * (j + 1) // 2
        s = r[idx_jj] - s
        
        if s <= 0.0:
            info = j + 1
            return r, info
        
        r[idx_jj] = np.sqrt(s)
    
    return r, info


def r8pp_sl(n, r_factor, b):
    """
    求解 packed Cholesky 分解后的线性系统 —— 基于 r8pp_sl.m
    
    解方程: R'·R·x = b
    
    Parameters
    ----------
    n : int
        矩阵阶数
    r_factor : ndarray
        r8pp_fa 输出的 R 因子
    b : ndarray, shape (n,)
        右端项
    
    Returns
    -------
    x : ndarray, shape (n,)
        解向量
    """
    if n < 1:
        raise ValueError("矩阵阶数必须 >= 1")
    b = np.asarray(b, dtype=float)
    if b.shape != (n,):
        raise ValueError("b 的形状必须与 n 一致")
    
    x = b.copy()
    
    # 前向代换: R'·y = b
    for k in range(n):
        t = 0.0
        for i in range(k):
            idx_ik = i + k * (k + 1) // 2
            t += r_factor[idx_ik] * x[i]
        idx_kk = k + k * (k + 1) // 2
        if abs(r_factor[idx_kk]) < 1e-15:
            x[k] = 0.0
        else:
            x[k] = (x[k] - t) / r_factor[idx_kk]
    
    # 后向代换: R·x = y
    for k in range(n - 1, -1, -1):
        idx_kk = k + k * (k + 1) // 2
        if abs(r_factor[idx_kk]) < 1e-15:
            x[k] = 0.0
        else:
            x[k] /= r_factor[idx_kk]
        t = -x[k]
        for i in range(k):
            idx_ik = i + k * (k + 1) // 2
            x[i] += t * r_factor[idx_ik]
    
    return x


def r8pp_mv(n, a, x_vec):
    """
    packed 对称矩阵与向量相乘 —— 基于 r8pp_mv.m
    
    Parameters
    ----------
    n : int
        矩阵阶数
    a : ndarray
        packed 存储的矩阵
    x_vec : ndarray, shape (n,)
        向量
    
    Returns
    -------
    b : ndarray, shape (n,)
        乘积 A·x
    """
    x_vec = np.asarray(x_vec, dtype=float)
    b = np.zeros(n, dtype=float)
    
    for i in range(n):
        for j in range(i):
            k = j + (i * (i + 1)) // 2
            b[i] += a[k] * x_vec[j]
        for j in range(i, n):
            k = i + (j * (j + 1)) // 2
            b[i] += a[k] * x_vec[j]
    
    return b


# =============================================================================
# 基于 451_gauss_seidel 的迭代求解器
# =============================================================================

def gauss_seidel_step(n, A, b, x):
    """
    单步 Gauss-Seidel 迭代 —— 基于 gauss_seidel1.m
    
    更新公式:
        x_i^{new} = (b_i - Σ_{j<i} A_{ij} x_j^{new} - Σ_{j>i} A_{ij} x_j^{old}) / A_{ii}
    
    Parameters
    ----------
    n : int
        矩阵维数
    A : ndarray, shape (n, n)
        系数矩阵
    b : ndarray, shape (n,)
        右端项
    x : ndarray, shape (n,)
        当前解估计
    
    Returns
    -------
    x_new : ndarray, shape (n,)
        新解估计
    """
    x_new = np.zeros(n, dtype=float)
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise ValueError(f"对角元 A[{i},{i}] 接近零，GS 迭代不收敛")
        x_new[i] = b[i]
        x_new[i] -= np.dot(A[i, :i], x_new[:i])
        x_new[i] -= np.dot(A[i, i + 1:], x[i + 1:])
        x_new[i] /= A[i, i]
    return x_new


def gauss_seidel_solve(A, b, x0=None, tol=1e-10, max_iter=10000):
    """
    Gauss-Seidel 迭代求解线性系统 A·x = b
    
    收敛判据:
        ||A·x - b||₂ / ||b||₂ < tol
    
    Parameters
    ----------
    A : ndarray
        系数矩阵 (建议对角占优)
    b : ndarray
        右端项
    x0 : ndarray, optional
        初始猜测
    tol : float
        相对残差容差
    max_iter : int
        最大迭代次数
    
    Returns
    -------
    x : ndarray
        数值解
    residual_history : list
        残差收敛历史
    """
    n = len(b)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        b_norm = 1.0
    
    residual_history = []
    for it in range(max_iter):
        x = gauss_seidel_step(n, A, b, x)
        res = np.linalg.norm(A.dot(x) - b) / b_norm
        residual_history.append(res)
        if res < tol:
            break
    
    return x, residual_history


# =============================================================================
# 基于 1154_st_to_ccs 的稀疏矩阵格式转换
# =============================================================================

def st_to_ccs_size(nst, ist, jst):
    """
    计算 ST→CCS 转换后的非零元数量 —— 基于 st_to_ccs_size.m
    
    Parameters
    ----------
    nst : int
        ST 格式非零元数量
    ist, jst : ndarray
        ST 格式的行、列索引
    
    Returns
    -------
    ncc : int
        CCS 格式非零元数量
    """
    if nst == 0:
        return 0
    pairs = set()
    for k in range(nst):
        pairs.add((ist[k], jst[k]))
    return len(pairs)


def st_to_ccs_index(nst, ist, jst, ncc, n):
    """
    生成 CCS 格式的索引数组 —— 基于 st_to_ccs_index.m
    
    Parameters
    ----------
    nst : int
        ST 格式非零元数量
    ist, jst : ndarray
        ST 行、列索引
    ncc : int
        CCS 非零元数量
    n : int
        矩阵列数
    
    Returns
    -------
    icc : ndarray
        CCS 行索引
    ccc : ndarray
        CCS 列压缩指针 (长度 n+1)
    """
    if nst == 0:
        return np.array([], dtype=int), np.zeros(n + 1, dtype=int)
    
    # 按列、行排序
    data = []
    for k in range(nst):
        data.append((jst[k], ist[k]))
    data = sorted(set(data))
    
    icc = np.array([row for _, row in data], dtype=int)
    jcc = np.array([col for col, _ in data], dtype=int)
    
    ccc = np.zeros(n + 1, dtype=int)
    ccc[0] = 0
    jlo = 0
    for i in range(ncc):
        jhi = jcc[i]
        if jhi != jlo:
            ccc[jlo + 1:jhi + 1] = i
            jlo = jhi
    ccc[jlo + 1:] = ncc
    
    return icc, ccc


def st_to_ccs_values(nst, ist, jst, ast, ncc, n, icc, ccc):
    """
    转换 ST 数值到 CCS 格式 —— 基于 st_to_ccs_values.m
    
    Parameters
    ----------
    nst : int
        ST 非零元数
    ist, jst, ast : ndarray
        ST 行、列、值
    ncc : int
        CCS 非零元数
    n : int
        列数
    icc, ccc : ndarray
        CCS 索引
    
    Returns
    -------
    acc : ndarray
        CCS 值数组
    """
    acc = np.zeros(ncc, dtype=float)
    
    for kst in range(nst):
        i = ist[kst]
        j = jst[kst]
        clo = ccc[j]
        chi = ccc[j + 1]
        
        found = False
        for kcc in range(clo, chi):
            if icc[kcc] == i:
                acc[kcc] += ast[kst]
                found = True
                break
        
        if not found:
            raise ValueError(f"ST 条目 ({i},{j}) 无法在 CCS 数组中定位")
    
    return acc


def ccs_mv(n, icc, ccc, acc, x_vec):
    """
    CCS 稀疏矩阵与向量相乘
    
    Parameters
    ----------
    n : int
        矩阵维度
    icc, ccc, acc : ndarray
        CCS 格式数组
    x_vec : ndarray
        向量
    
    Returns
    -------
    b : ndarray
        乘积 A·x
    """
    b = np.zeros(n, dtype=float)
    for j in range(n):
        for kcc in range(ccc[j], ccc[j + 1]):
            i = icc[kcc]
            b[i] += acc[kcc] * x_vec[j]
    return b


# =============================================================================
# 基于 1085_sine_transform 的谱方法
# =============================================================================

def sine_transform_data(n, f_vals):
    """
    对离散数据做离散正弦变换 —— 基于 sine_transform_data.m
    
    变换公式:
        S(k) = √(2/(N+1)) Σ_{j=1}^{N} sin(π·k·j/(N+1)) · f(j)
    
    Parameters
    ----------
    n : int
        数据点数
    f_vals : ndarray, shape (n,)
        定义在内部节点上的函数值
    
    Returns
    -------
    s : ndarray, shape (n,)
        正弦变换系数
    """
    f_vals = np.asarray(f_vals, dtype=float)
    if len(f_vals) != n:
        raise ValueError("f_vals 长度必须等于 n")
    
    s = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            s[i] += np.sin(np.pi * (i + 1) * (j + 1) / (n + 1)) * f_vals[j]
    s *= np.sqrt(2.0 / (n + 1))
    return s


def sine_transform_interpolant(n, a, b, s_coeffs, x_query):
    """
    由正弦变换系数重构插值函数 —— 基于 sine_transform_interpolant.m
    
    插值公式:
        f(x) = f₁(x) + Σ_{k=1}^{N} S(k)·sin(π·k·(x-a)/(b-a)) · √(2/(N+1))
    
    其中 f₁(x) 是通过端点的线性函数（此处假设为零）。
    
    Parameters
    ----------
    n : int
        系数数量
    a, b : float
        区间端点
    s_coeffs : ndarray
        正弦变换系数
    x_query : float
        查询点
    
    Returns
    -------
    float
        插值结果
    """
    if not (a <= x_query <= b):
        raise ValueError("查询点必须在 [a, b] 区间内")
    if abs(b - a) < 1e-15:
        raise ValueError("区间长度必须为正")
    
    # 纯正弦展开，需包含归一化因子
    norm = np.sqrt(2.0 / (n + 1))
    f_interp = 0.0
    for k in range(n):
        f_interp += s_coeffs[k] * np.sin(np.pi * (k + 1) * (x_query - a) / (b - a)) * norm
    
    return f_interp


# =============================================================================
# 平面波展开 (PWE) 本征求解器
# =============================================================================

def build_pwe_matrix(n_g, eps_r, a, kx, ky):
    """
    构建二维 TE 模式平面波展开哈密顿量矩阵
    
    对于 TE 模式，本征方程为:
        Σ_G' κ(G-G') |k+G'|² E(k+G') = (ω²/c²) E(k+G)
    
    其中 κ(G) 为介电函数倒易空间的傅里叶系数:
        κ(G) = (1/V_cell) ∫_{cell} (1/ε(r)) e^{-iG·r} d²r
    
    Parameters
    ----------
    n_g : int
        每个方向平面波数量 (总平面波数 = (2n_g+1)²)
    eps_r : ndarray
        实空间介电常数分布
    a : float
        晶格常数
    kx, ky : float
        布洛赫波矢
    
    Returns
    -------
    H : ndarray, shape (N_pw, N_pw)
        哈密顿量矩阵
    G_vec : ndarray
        倒格矢列表
    """
    nx, ny = eps_r.shape
    N_pw = (2 * n_g + 1) ** 2
    
    # 生成倒格矢 (正方晶格)
    G_vec = []
    for i in range(-n_g, n_g + 1):
        for j in range(-n_g, n_g + 1):
            G_vec.append([i * 2 * np.pi / a, j * 2 * np.pi / a])
    G_vec = np.array(G_vec)
    
    # 计算 1/ε(r) 的傅里叶变换
    inv_eps = 1.0 / np.maximum(eps_r, 1e-12)
    kappa_G = np.fft.fft2(inv_eps) / (nx * ny)
    
    H = np.zeros((N_pw, N_pw), dtype=complex)
    
    for m in range(N_pw):
        for n in range(N_pw):
            dG = G_vec[m] - G_vec[n]
            # 找到对应的 FFT 索引
            ig = int(np.round(dG[0] * a / (2 * np.pi))) % nx
            jg = int(np.round(dG[1] * a / (2 * np.pi))) % ny
            kappa = kappa_G[ig, jg]
            
            k_plus_G = np.array([kx, ky]) + G_vec[n]
            k_mag2 = k_plus_G[0] ** 2 + k_plus_G[1] ** 2
            
            H[m, n] = kappa * k_mag2
    
    return H, G_vec


def solve_bands_pwe(n_bands, n_g, eps_r, a, k_points):
    """
    用平面波展开法求解光子晶体能带
    
    求解广义本征值问题:
        H·E = (ω²/c²)·E
    
    Parameters
    ----------
    n_bands : int
        求解能带数量
    n_g : int
        平面波截断
    eps_r : ndarray
        介电常数分布
    a : float
        晶格常数
    k_points : ndarray, shape (N_k, 2)
        k 点路径
    
    Returns
    -------
    omega_bands : ndarray, shape (N_k, n_bands)
        各 k 点处的本征频率 [rad/s]
    """
    N_k = len(k_points)
    omega_bands = np.zeros((N_k, n_bands))
    
    for ik, (kx, ky) in enumerate(k_points):
        H, _ = build_pwe_matrix(n_g, eps_r, a, kx, ky)
        # 转换为实对称矩阵 (对于 TE 模式 H 是厄米的)
        H_sym = 0.5 * (H + H.conj().T)
        
        # 数值稳定性: 对接近零的特征值进行正则化
        H_sym += 1e-14 * np.eye(H_sym.shape[0])
        
        eigenvalues = np.linalg.eigvalsh(H_sym)
        eigenvalues = np.sort(np.real(eigenvalues))
        
        # ω = c·√λ
        for ib in range(min(n_bands, len(eigenvalues))):
            lam = max(eigenvalues[ib], 0.0)
            omega_bands[ik, ib] = C_0 * np.sqrt(lam)
    
    return omega_bands


# =============================================================================
# 基于离散正弦变换的谱方法求解器 (一维层状结构)
# =============================================================================

def solve_layered_structure_spectral(n_modes, n_pts, eps_profile, a, kx):
    """
    用离散正弦变换谱方法求解一维层状光子晶体
    
    对于 y 方向无限延伸的层状结构，场量可展开为:
        H_z(x,y) = Σ_n c_n sin(nπx/a) e^{i(k_x y - ωt)}
    
    代入波动方程得到本征值问题:
        Σ_m K_{nm} c_m = (ω²/c²) c_n
    
    其中:
        K_{nm} = (2/a) ∫_0^a sin(nπx/a) (1/ε(x)) [-(d²/dx²)+k_x²] sin(mπx/a) dx
    
    Parameters
    ----------
    n_modes : int
        正弦模态数量
    n_pts : int
        空间离散点数
    eps_profile : ndarray, shape (n_pts,)
        沿 x 方向的介电常数分布
    a : float
        结构周期 [m]
    kx : float
        平行于层方向的波矢 [m⁻¹]
    
    Returns
    -------
    omega : ndarray, shape (n_modes,)
        本征频率
    modes : ndarray, shape (n_modes, n_pts)
        本征模态
    """
    if n_modes > n_pts:
        raise ValueError("模态数不能超过空间点数")
    
    dx = a / (n_pts + 1)
    x = np.linspace(dx, a - dx, n_pts)
    
    # 计算核矩阵 K
    K = np.zeros((n_modes, n_modes), dtype=float)
    
    for n in range(n_modes):
        for m in range(n_modes):
            # 对每一对 (n,m)，数值积分
            integrand = np.zeros(n_pts)
            for i in range(n_pts):
                sin_n = np.sin((n + 1) * np.pi * x[i] / a)
                sin_m = np.sin((m + 1) * np.pi * x[i] / a)
                # 正弦函数的二阶导数: d²/dx² sin(mπx/a) = -(mπ/a)² sin(mπx/a)
                laplacian_m = ((m + 1) * np.pi / a) ** 2 * sin_m + kx ** 2 * sin_m
                integrand[i] = sin_n * (1.0 / max(eps_profile[i], 1e-12)) * laplacian_m
            
            K[n, m] = (2.0 / a) * np.trapz(integrand, x)
    
    # 对称化
    K = 0.5 * (K + K.T)
    
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))
    
    omega = C_0 * np.sqrt(eigenvalues)
    modes = eigenvectors.T
    
    return omega, modes
