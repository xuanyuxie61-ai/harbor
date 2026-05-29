"""
poisson_solver.py
背景电磁场的泊松型方程求解器

基于 606_jacobi_poisson_1d 项目重构:
  有限差分离散 + Jacobi 迭代求解
  
物理应用:
  在希格斯衰变分析中，探测器内带电粒子产生的电磁 showers
  可用泊松方程描述电势分布:
    -nabla^2 phi = rho / epsilon_0
  
  在 1D 简化模型中 (径向对称):
    -(1/r^2) * d/dr(r^2 * dphi/dr) = rho(r)
    
  或者更简单地，在有限区间上:
    -d^2u/dx^2 = f(x),  u(0)=u(1)=0
    
  这里我们实现后者，并用于背景场的平滑化处理。
"""
import numpy as np
from constants import TINY, MAX_ITER

# ============================================================
# 1. 有限差分 Laplacian 构造 (映射 606_jacobi_poisson_1d)
# ============================================================
def build_fd_laplacian_1d(n, h):
    """
    构造 1D 有限差分 Laplacian 矩阵 (Dirichlet 边界)
    
    离散格式 (二阶中心差分):
      -u''(x_i) ~ (-u_{i-1} + 2*u_i - u_{i+1}) / h^2
    
    矩阵形式 A u = f，其中 A 为三对角矩阵:
      A_{i,i} = 2/h^2
      A_{i,i+1} = A_{i,i-1} = -1/h^2
    
    参数:
        n: 内部节点数
        h: 网格间距
    返回:
        A: 稀疏矩阵 (dense numpy array for simplicity)
    """
    A = np.zeros((n, n))
    inv_h2 = 1.0 / (h * h)
    for i in range(n):
        A[i, i] = 2.0 * inv_h2
        if i > 0:
            A[i, i - 1] = -inv_h2
        if i < n - 1:
            A[i, i + 1] = -inv_h2
    return A


# ============================================================
# 2. Jacobi 迭代求解器 (映射 606_jacobi_poisson_1d)
# ============================================================
def jacobi_solve(A, f, u0=None, max_iter=MAX_ITER, tol=1.0e-10):
    """
    Jacobi 迭代求解 Au = f
    
    迭代格式:
      u_i^{(k+1)} = (f_i - sum_{j!=i} A_{ij} * u_j^{(k)}) / A_{ii}
    
    收敛条件: 谱半径 rho(D^{-1}(L+U)) < 1
    对 Laplacian 矩阵，Jacobi 方法收敛但较慢 (Gauss-Seidel/SSOR 更快)
    
    参数:
        A: 系数矩阵
        f: 右端项
        u0: 初始猜测
        max_iter: 最大迭代次数
        tol: 残差容差
    返回:
        u: 近似解
        residual_norm: 最终残差范数
        converged: 是否收敛
        iterations: 实际迭代次数
    """
    n = A.shape[0]
    f = np.asarray(f, dtype=float)
    
    if u0 is None:
        u = np.zeros(n)
    else:
        u = np.asarray(u0, dtype=float).copy()
    
    diag = np.diag(A).copy()
    if np.any(np.abs(diag) < TINY):
        raise ValueError("Zero diagonal element in Jacobi iteration")
    
    for it in range(max_iter):
        u_new = np.zeros(n)
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * u[j]
            u_new[i] = (f[i] - sigma) / diag[i]
        
        # 残差计算
        residual = f - A @ u_new
        res_norm = np.linalg.norm(residual) / np.sqrt(n)
        
        u = u_new
        
        if res_norm < tol:
            return u, res_norm, True, it + 1
    
    residual = f - A @ u
    res_norm = np.linalg.norm(residual) / np.sqrt(n)
    return u, res_norm, False, max_iter


# ============================================================
# 3. 带松弛因子的 Jacobi (SOR 风格)
# ============================================================
def sor_solve(A, f, omega=1.5, u0=None, max_iter=MAX_ITER, tol=1.0e-10):
    """
    Successive Over-Relaxation (SOR) 求解器
    
    Gauss-Seidel 更新:
      u_i^{new} = (f_i - sum_{j<i} A_{ij} u_j^{new} - sum_{j>i} A_{ij} u_j^{old}) / A_{ii}
    
    SOR 加速:
      u_i^{new} = (1-omega) * u_i^{old} + omega * u_i^{GS}
    
    最优 omega 对 1D Laplacian: omega_opt = 2 / (1 + sin(pi/(n+1)))
    
    参数:
        A: 系数矩阵
        f: 右端项
        omega: 松弛因子 (1.0 = Gauss-Seidel)
        u0: 初始猜测
        max_iter, tol: 控制参数
    返回:
        u, residual_norm, converged, iterations
    """
    n = A.shape[0]
    f = np.asarray(f, dtype=float)
    
    if u0 is None:
        u = np.zeros(n)
    else:
        u = np.asarray(u0, dtype=float).copy()
    
    diag = np.diag(A)
    if np.any(np.abs(diag) < TINY):
        raise ValueError("Zero diagonal element in SOR")
    
    for it in range(max_iter):
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * u[j]
            u_gs = (f[i] - sigma) / diag[i]
            u[i] = (1.0 - omega) * u[i] + omega * u_gs
        
        residual = f - A @ u
        res_norm = np.linalg.norm(residual) / np.sqrt(n)
        if res_norm < tol:
            return u, res_norm, True, it + 1
    
    residual = f - A @ u
    res_norm = np.linalg.norm(residual) / np.sqrt(n)
    return u, res_norm, False, max_iter


# ============================================================
# 4. 直接求解器 (用于验证)
# ============================================================
def direct_solve(A, f):
    """
    使用 numpy 的线性求解器 (LU 分解) 直接求解
    用于验证 Jacobi/SOR 的正确性
    """
    return np.linalg.solve(A, f)


# ============================================================
# 5. 背景场平滑化 (泊松正则化)
# ============================================================
def smooth_background_poisson(raw_counts, smoothing_strength=1.0, n_inner=50):
    """
    使用泊松方程对背景计数进行平滑化/正则化
    
    模型:
      -alpha * u''(x) + u(x) = raw(x)
      u(0) = raw(0), u(1) = raw(1)
    
    这相当于加了扩散正则项，抑制高频噪声。
    
    参数:
        raw_counts: 原始计数数组
        smoothing_strength: alpha，越大越平滑
        n_inner: 内部节点数
    返回:
        smoothed: 平滑后的数组
    """
    n = len(raw_counts)
    if n < 3:
        return raw_counts.copy()
    
    # 在内部节点上离散
    h = 1.0 / (n_inner + 1)
    A = build_fd_laplacian_1d(n_inner, h)
    A = smoothing_strength * A + np.eye(n_inner)
    
    # 右端项: 插值原始数据
    x_inner = np.linspace(h, 1.0 - h, n_inner)
    x_raw = np.linspace(0.0, 1.0, n)
    f_interp = np.interp(x_inner, x_raw, raw_counts)
    
    # Dirichlet 边界贡献移到右端
    f_interp[0] += smoothing_strength * raw_counts[0] / (h * h)
    f_interp[-1] += smoothing_strength * raw_counts[-1] / (h * h)
    
    # 求解
    try:
        u_inner = direct_solve(A, f_interp)
    except np.linalg.LinAlgError:
        u_inner, _, _, _ = sor_solve(A, f_interp, omega=1.2)
    
    # 组合边界和内部
    smoothed = np.zeros(n)
    smoothed[0] = raw_counts[0]
    smoothed[-1] = raw_counts[-1]
    smoothed[1:-1] = np.interp(x_raw[1:-1], x_inner, u_inner)
    
    return smoothed


# ============================================================
# 6. 误差分析
# ============================================================
def compare_solvers(f_func, exact_func, n=31):
    """
    对比不同求解器在 1D Poisson 方程上的误差
    
    方程: -u'' = f(x), u(0)=u(1)=0
    """
    h = 1.0 / (n + 1)
    x = np.linspace(h, 1.0 - h, n)
    A = build_fd_laplacian_1d(n, h)
    f = np.array([f_func(xi) for xi in x])
    u_exact = np.array([exact_func(xi) for xi in x])
    
    # 直接求解
    u_direct = direct_solve(A, f)
    err_direct = np.linalg.norm(u_direct - u_exact) / np.linalg.norm(u_exact)
    
    # Jacobi
    u_jacobi, _, conv_j, it_j = jacobi_solve(A, f, max_iter=20000, tol=1.0e-12)
    err_jacobi = np.linalg.norm(u_jacobi - u_exact) / np.linalg.norm(u_exact)
    
    # SOR
    omega_opt = 2.0 / (1.0 + np.sin(np.pi / (n + 1)))
    u_sor, _, conv_s, it_s = sor_solve(A, f, omega=omega_opt, max_iter=10000, tol=1.0e-12)
    err_sor = np.linalg.norm(u_sor - u_exact) / np.linalg.norm(u_exact)
    
    return {
        "direct_error": err_direct,
        "jacobi_error": err_jacobi,
        "jacobi_converged": conv_j,
        "jacobi_iters": it_j,
        "sor_error": err_sor,
        "sor_converged": conv_s,
        "sor_iters": it_s,
        "omega_opt": omega_opt,
    }
