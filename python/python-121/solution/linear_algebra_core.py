"""
linear_algebra_core.py
线性代数核心求解模块

融入原项目:
- 988_r8pbu: 对称正定带状矩阵的共轭梯度法
- 902_power_method: 幂法求主特征值
- 338_errors: 数值误差分析

功能:
1. 共轭梯度法求解大型稀疏对称正定线性系统
2. 幂法分析心脏电传播稳定性特征值
3. 矩阵条件数与数值稳定性分析
"""

import numpy as np
from math import sqrt


# ============================================================================
# 带状矩阵向量乘法（源自 988_r8pbu/r8pbu_mv）
# ============================================================================

def r8pbu_mv(n, mu, a, x):
    """
    对称正定带状矩阵与向量相乘 y = A*x
    
    带状存储格式:
    - 对角线存储在第 mu+1 行
    - 第 k 条上对角线存储在第 mu+1-k 行，列从 k+1 到 n
    
    参数:
        n: 矩阵阶数
        mu: 上带宽
        a: (mu+1, n) 带状存储矩阵
        x: n维向量
    返回:
        y = A*x
    """
    if n <= 0 or mu < 0 or mu > n - 1:
        return np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        # 对角元
        y[i] += a[mu, i] * x[i]
        # 上三角元素（利用对称性）
        for j in range(i + 1, min(i + mu + 1, n)):
            a_val = a[mu + i - j, j]
            y[i] += a_val * x[j]
            y[j] += a_val * x[i]
    return y


def r8pbu_cg(n, mu, a, b, x0, tol=1e-12, max_iter=None):
    """
    共轭梯度法求解 A*x = b
    
    算法推导:
    1. 初始化: r_0 = b - A*x_0, p_0 = r_0
    2. 迭代 k=0,1,2,...:
       α_k = (r_k^T r_k) / (p_k^T A p_k)
       x_{k+1} = x_k + α_k p_k
       r_{k+1} = r_k - α_k A p_k
       β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
       p_{k+1} = r_{k+1} + β_k p_k
    
    理论性质:
    - 对于n维问题，CG理论上最多n步收敛（精确算术）
    - 实际收敛速率取决于条件数 κ(A):
      ||e_k||_A / ||e_0||_A ≤ 2 * ((sqrt(κ)-1)/(sqrt(κ)+1))^k
    
    参数:
        n, mu, a: 带状矩阵参数
        b: 右端项
        x0: 初始猜测
        tol: 残差容差
        max_iter: 最大迭代次数
    返回:
        x: 近似解
        residual_norm: 最终残差范数
        iter_count: 实际迭代次数
    """
    if max_iter is None:
        max_iter = n
    
    b = np.asarray(b, dtype=float).flatten()
    x = np.asarray(x0, dtype=float).flatten().copy()
    
    if n <= 0 or mu < 0:
        return x, float('inf'), 0
    
    # 初始残差 r = b - A*x
    ap = r8pbu_mv(n, mu, a, x)
    r = b - ap
    p = r.copy()
    
    rs_old = np.dot(r, r)
    rs0 = rs_old
    
    if rs0 == 0.0:
        return x, 0.0, 0
    
    for it in range(1, max_iter + 1):
        ap = r8pbu_mv(n, mu, a, p)
        pap = np.dot(p, ap)
        
        if pap == 0.0:
            break
        
        alpha = rs_old / pap
        x += alpha * p
        r -= alpha * ap
        
        rs_new = np.dot(r, r)
        residual_norm = sqrt(rs_new)
        
        if residual_norm / sqrt(rs0) < tol:
            return x, residual_norm, it
        
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    
    return x, sqrt(rs_old), max_iter


def build_laplacian_banded(nx, ny, dx, dy):
    """
    构建2D拉普拉斯算子的带状对称正定矩阵（五点差分格式）
    
    离散方程:
    ∇²u ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/dx² 
        + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/dy²
    
    对于均匀网格 (dx=dy=h):
    ∇²u ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²
    
    矩阵带宽 μ = nx（因为 y 方向相邻节点索引差为 nx）
    """
    n = nx * ny
    mu = nx
    a = np.zeros((mu + 1, n))
    
    h2 = dx * dy
    if h2 == 0:
        h2 = 1.0
    
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            # 对角元
            diag_val = 0.0
            if i > 0:
                diag_val += 1.0 / dx ** 2
            if i < nx - 1:
                diag_val += 1.0 / dx ** 2
            if j > 0:
                diag_val += 1.0 / dy ** 2
            if j < ny - 1:
                diag_val += 1.0 / dy ** 2
            a[mu, idx] = diag_val
            
            # x方向邻居（上对角线，距离1）
            if i < nx - 1:
                a[mu - 1, idx + 1] = -1.0 / dx ** 2
            
            # y方向邻居（上对角线，距离nx）
            if j < ny - 1:
                a[mu - nx, idx + nx] = -1.0 / dy ** 2
    
    return n, mu, a


# ============================================================================
# 幂法求主特征值（源自 902_power_method）
# ============================================================================

def power_method(A, y0, it_max=1000, tol=1e-10):
    """
    幂法计算矩阵主特征值及对应特征向量
    
    算法:
    1. 归一化初始向量 y_0
    2. 迭代: 
       z_k = A * y_{k-1}
       λ_k = y_{k-1}^T * z_k  (Rayleigh商)
       y_k = z_k / ||z_k||
    
    收敛条件:
    |λ_k - λ_{k-1}| < tol
    
    收敛速率:
    |λ^{(k)} - λ_1| = O(|λ_2/λ_1|^k)
    其中 λ_1, λ_2 分别是最大和次大特征值（按模）
    
    参数:
        A: N×N 矩阵
        y0: 初始向量
        it_max: 最大迭代次数
        tol: 特征值收敛容差
    返回:
        y: 特征向量估计
        lambda_val: 特征值估计
        it_num: 实际迭代次数
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y0, dtype=float).flatten().copy()
    n = A.shape[0]
    
    if n == 0:
        return y, 0.0, 0
    
    norm_y = np.linalg.norm(y)
    if norm_y == 0:
        y = np.ones(n)
        norm_y = sqrt(n)
    y = y / norm_y
    
    ay = A.dot(y)
    lambda_val = np.dot(y, ay)
    y = ay / np.linalg.norm(ay)
    if lambda_val < 0:
        y = -y
    
    for it_num in range(1, it_max + 1):
        lambda_old = lambda_val
        y_old = y.copy()
        
        ay = A.dot(y)
        lambda_val = np.dot(y, ay)
        norm_ay = np.linalg.norm(ay)
        if norm_ay == 0:
            break
        y = ay / norm_ay
        if lambda_val < 0:
            y = -y
        
        val_dif = abs(lambda_val - lambda_old)
        
        # 特征向量方向收敛判断
        cos_yy = np.dot(y, y_old)
        sin_yy = sqrt(max(0.0, (1.0 - cos_yy) * (1.0 + cos_yy)))
        
        if val_dif <= tol and sin_yy <= tol:
            # 归一化特征向量使得 A*y = lambda*y
            y = ay / lambda_val if lambda_val != 0 else y
            return y, lambda_val, it_num
    
    if lambda_val != 0:
        y = ay / lambda_val
    return y, lambda_val, it_max


def stability_eigenvalue_analysis(diffusion_matrix, reaction_jacobian):
    """
    心脏电传播稳定性特征值分析
    
    对于反应扩散系统 ∂u/∂t = D∇²u + f(u)
    线性化后的特征值问题:
    (D∇² + J_f) * φ = λ * φ
    
    稳定性判据:
    - 若所有 Re(λ) < 0: 系统稳定
    - 若存在 Re(λ) > 0: 系统不稳定，可能发生心律失常
    
    参数:
        diffusion_matrix: 扩散算子离散矩阵
        reaction_jacobian: 反应项Jacobian矩阵
    返回:
        lambda_max: 最大实部特征值
        is_stable: 是否稳定
    """
    A = diffusion_matrix + reaction_jacobian
    # 使用幂法估计主特征值（假设矩阵对称或近似对称）
    n = A.shape[0]
    y0 = np.random.randn(n)
    _, lambda_max, _ = power_method(A, y0, it_max=min(n, 500), tol=1e-8)
    
    is_stable = lambda_max < 0
    return lambda_max, is_stable


# ============================================================================
# 高级求解器封装
# ============================================================================

def solve_poisson_2d_cg(f, nx, ny, dx, dy, boundary_value=0.0, max_iter=None):
    """
    使用共轭梯度法求解2D泊松方程:
    ∇²φ = f
    
    离散形式:
    (φ_{i+1,j} - 2φ_{i,j} + φ_{i-1,j})/dx² 
    + (φ_{i,j+1} - 2φ_{i,j} + φ_{i,j-1})/dy² = f_{i,j}
    
    参数:
        f: (nx, ny) 右端项
        nx, ny: 网格数
        dx, dy: 网格间距
        boundary_value: 边界值（Dirichlet条件）
    返回:
        phi: (nx, ny) 解
    """
    n = nx * ny
    n_total, mu, a = build_laplacian_banded(nx, ny, dx, dy)
    
    b = np.asarray(f, dtype=float).flatten()
    x0 = np.zeros(n)
    
    # Dirichlet边界处理：简化处理，边界值固定
    # 这里假设边界已在f中处理
    
    phi_flat, res, iters = r8pbu_cg(n_total, mu, a, b, x0, tol=1e-10, max_iter=max_iter)
    phi = phi_flat.reshape((nx, ny))
    return phi, res, iters
