"""
modal_analysis.py
室内声场模态分析与特征频率求解
基于 zero_rc (Brent 反向通信求根) 与 ball_integrals (球面积分) 核心算法重构

声学工程应用：
封闭空间的声学模态满足 Helmholtz 方程的特征值问题：
    ∇² φ + λ φ = 0,   λ = (ω/c)²
在刚性壁面边界条件（∂φ/∂n = 0）下，对于长方体房间有解析解：
    f_{l,m,n} = (c/2) * √((l/Lx)² + (m/Ly)² + (n/Lz)²)

本模块实现：
1. 解析模态频率计算（用于验证）
2. 基于 FEM 质量/刚度矩阵的广义特征值问题求解（逆迭代法）
3. 基于 Brent 方法的特征频率精细化搜索（zero_rc 思想）
4. 模态参与因子与球面积分计算
"""

import numpy as np
from fem_acoustics import C_AIR


def rectangular_room_modes(Lx, Ly, Lz, max_order=5):
    """
    计算长方体房间的解析模态频率：
        f_{l,m,n} = (c/2) * sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2)
    边界条件：刚性壁（Neumann），l,m,n ∈ [0, max_order]。
    """
    modes = []
    for l in range(max_order + 1):
        for m in range(max_order + 1):
            for n in range(max_order + 1):
                if l == 0 and m == 0 and n == 0:
                    continue
                f = (C_AIR / 2.0) * np.sqrt(
                    (l / Lx) ** 2 + (m / Ly) ** 2 + (n / Lz) ** 2
                )
                modes.append({
                    'l': l, 'm': m, 'n': n,
                    'frequency': f,
                    'wavelength': C_AIR / f if f > 0 else np.inf
                })
    modes.sort(key=lambda x: x['frequency'])
    return modes


def schroeder_frequency(room_volume, total_surface_area, absorption_coeff_avg):
    """
    Schroeder 频率（临界频率）：低于此频率以模态响应为主，
    高于此频率以扩散声场为主。
        f_s = 2000 * sqrt(T60 / V)   [Hz]
    或使用统计公式：
        f_s = c/2 * sqrt(A / (V * ln(10)))
    其中 A 为等效吸声面积。
    """
    A_eq = total_surface_area * absorption_coeff_avg
    if A_eq < 1e-14:
        A_eq = 1e-14
    # 简化的 Schroeder 频率估算
    f_s = 2000.0 * np.sqrt(absorption_coeff_avg * total_surface_area / room_volume)
    return f_s


def zero_rc_brent(func, a, b, tol=1e-10, max_iter=100):
    """
    Brent 求根法的简化实现（非反向通信版，用于模态频率搜索）。
    基于 zero_rc 的 Dekker-Brent 混合策略：
    优先逆二次插值，否则退化为二分法或割线法。

    求特征方程 det(K - λ M) = 0 的根。
    """
    fa = func(a)
    fb = func(b)
    if fa * fb > 0:
        raise ValueError("Root not bracketed: f(a) and f(b) must have opposite signs")

    c, fc = a, fa
    d = e = b - a

    for _ in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2.0 * np.finfo(float).eps * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        if abs(m) <= tol_act or abs(fb) < tol:
            return b
        if abs(e) < tol_act or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:
                # 割线法
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # 逆二次插值
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            min1 = 3.0 * m * q - abs(tol_act * q)
            min2 = abs(e * q)
            if 2.0 * p < min(min1, min2):
                e = d
                d = p / q
            else:
                d = e = m
        a, fa = b, fb
        if abs(d) > tol_act:
            b += d
        else:
            b += np.sign(m) * tol_act
        fb = func(b)
    return b


def inverse_iteration(K_sparse, M_sparse, max_iter=50, tol=1e-10):
    """
    逆迭代法求最小特征值与特征向量：
        K φ = λ M φ
    迭代格式：
        K x_{k+1} = M x_k
        x_{k+1} = x_{k+1} / ||x_{k+1}||_M
        λ_{k+1} = x_k^T K x_k / (x_k^T M x_k)
    """
    n = K_sparse.n
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)

    from sparse_linalg import conjugate_gradient

    for iteration in range(max_iter):
        b = M_sparse.mv(x)
        # 使用 CG 求解 K x_new = b
        x_new = conjugate_gradient(K_sparse, b, x0=x, tol=1e-8, max_iter=min(n, 500))
        # M-范数归一化
        Mx = M_sparse.mv(x_new)
        norm_m = np.sqrt(np.dot(x_new, Mx))
        if norm_m < 1e-14:
            break
        x_new = x_new / norm_m
        # Rayleigh 商
        Kx = K_sparse.mv(x_new)
        Mx = M_sparse.mv(x_new)
        lam = np.dot(x_new, Kx) / np.dot(x_new, Mx)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    # TODO (Hole 2): 从 Rayleigh 商 λ 计算特征频率
    # 提示：omega = sqrt(λ)，freq = omega * c / (2*pi)
    freq = 0.0  # FIXME: 需要正确的频率换算公式
    return freq, x, lam


def rayleigh_quotient_iteration(K_sparse, M_sparse, shift_guess, max_iter=20, tol=1e-12):
    """
    Rayleigh 商迭代：加速收敛到特定频率附近的模态。
    (K - μ M) x_{k+1} = M x_k
    μ_{k+1} = x_k^T K x_k / (x_k^T M x_k)
    """
    n = K_sparse.n
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    mu = shift_guess

    from sparse_linalg import conjugate_gradient, assemble_sparse_from_triplets

    for iteration in range(max_iter):
        # 构建平移矩阵 (K - mu M)
        A_rows, A_cols, A_vals = [], [], []
        for i in range(K_sparse.nnz):
            A_rows.append(K_sparse.rows[i])
            A_cols.append(K_sparse.cols[i])
            A_vals.append(K_sparse.vals[i])
        for i in range(M_sparse.nnz):
            A_rows.append(M_sparse.rows[i])
            A_cols.append(M_sparse.cols[i])
            A_vals.append(-mu * M_sparse.vals[i])
        A_shift = assemble_sparse_from_triplets(A_rows, A_cols, A_vals, n)
        # 添加小正则化保证正定性
        for i in range(n):
            A_rows.append(i)
            A_cols.append(i)
            A_vals.append(1e-8)
        A_shift = assemble_sparse_from_triplets(A_rows, A_cols, A_vals, n)

        b = M_sparse.mv(x)
        x_new = conjugate_gradient(A_shift, b, x0=x, tol=1e-7, max_iter=min(n, 500))
        Mx = M_sparse.mv(x_new)
        norm_m = np.sqrt(np.abs(np.dot(x_new, Mx)))
        if norm_m < 1e-14:
            break
        x_new = x_new / norm_m
        Kx = K_sparse.mv(x_new)
        Mx = M_sparse.mv(x_new)
        mu_new = np.dot(x_new, Kx) / np.dot(x_new, Mx)
        if abs(mu_new - mu) < tol:
            mu = mu_new
            x = x_new
            break
        mu = mu_new
        x = x_new

    omega = np.sqrt(abs(mu))
    freq = omega * C_AIR / (2.0 * np.pi)
    return freq, x, mu


def modal_participation_factor(mode_shape, force_dof):
    """
    模态参与因子：
        Γ_i = φ_i^T F / (φ_i^T M φ_i)
    衡量第 i 阶模态对声源激励的响应强度。
    """
    return mode_shape[force_dof]


def spherical_mode_integral(mode_shape, p, center, radius, n_samples=1000):
    """
    在球面上采样并计算模态振型的球面积分。
    基于 ball_integrals 的球面采样思想。
    用于评估模态在房间特定区域的能量集中度。
    """
    from quadrature_rules import ball01_sample
    # 在单位球内采样并归一化到球面
    samples = ball01_sample(n_samples)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-14)
    samples = samples / norms * radius + center

    # 找到最近的网格节点
    vals = []
    for s in samples:
        dists = np.linalg.norm(p - s, axis=1)
        idx = np.argmin(dists)
        vals.append(mode_shape[idx])

    # 球面积分近似：4πr² * mean
    integral = 4.0 * np.pi * radius ** 2 * np.mean(vals)
    return integral


def compute_modal_density(room_volume, freq):
    """
    模态密度：单位频率间隔内的模态数。
    对于三维矩形房间：
        n(f) = 4π V f² / c³ + π S f / (2 c²) + L / (8 c)
    其中 V 为体积，S 为总表面积，L 为总边长。
    """
    c = C_AIR
    # 简化的主项
    n_f = 4.0 * np.pi * room_volume * freq ** 2 / (c ** 3)
    return n_f


def modal_overlap_factor(modes, damping_ratio=0.01):
    """
    模态重叠因子（Modal Overlap Factor）：
        MOF = n(f) * η * f
    其中 η 为阻尼比。
    当 MOF > 1 时，声场进入扩散区。
    """
    mof_values = []
    for mode in modes:
        f = mode['frequency']
        n_f = compute_modal_density(400.0, f)  # 使用近似体积
        mof = n_f * damping_ratio * f
        mof_values.append({'frequency': f, 'mof': mof})
    return mof_values
