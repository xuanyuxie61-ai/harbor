"""
 helmholtz_solver.py
 
 融合种子项目:
   - 473_gmres: 广义最小残差法 (GMRES)、Arnoldi 迭代、Givens 旋转
 
 科学应用:
   频率域全波形反演的核心是求解 Helmholtz 方程:
     (nabla^2 + k^2(x)) u(x) = f(x)
   其中 k(x) = omega / c(x) 为波数，c(x) 为速度模型。
   
   在大规模问题中，直接求解计算量巨大，因此采用迭代方法。
   GMRES 是求解大型稀疏非对称线性系统的首选 Krylov 子空间方法，
   结合 Arnoldi 正交化和 Givens 平面旋转实现高效的残差最小化。
"""

import numpy as np


def givens_rotation(v1, v2):
    """
    计算 Givens 旋转矩阵参数。
    
    Givens 矩阵形式:
      G = [ cs  -sn ]
          [ sn   cs ]
    满足 G * [v1, v2]^T = [r, 0]^T。
    
    Parameters
    ----------
    v1, v2 : float
        待消去的向量分量。
    
    Returns
    -------
    cs, sn : float
        余弦和正弦值。
    """
    if abs(v1) < 1e-15:
        cs = 0.0
        sn = 1.0
    else:
        t = np.sqrt(v1 ** 2 + v2 ** 2)
        cs = abs(v1) / t
        sn = cs * v2 / v1
    return cs, sn


def apply_givens_rotation(h, cs, sn, k):
    """
    对已存储的 Givens 旋转序列应用到 H 的第 k 列。
    
    这是 GMRES 算法中维持上 Hessenberg 矩阵 R 为上三角的关键步骤。
    
    Parameters
    ----------
    h : ndarray, shape (k+1,)
        H 的第 k 列（将被修改）。
    cs : ndarray
        已存储的余弦值。
    sn : ndarray
        已存储的正弦值。
    k : int
        当前迭代步（1-based）。
    
    Returns
    -------
    h : ndarray
        修改后的列。
    cs_k, sn_k : float
        新的 Givens 旋转参数。
    """
    for i in range(k - 1):
        temp = cs[i] * h[i] + sn[i] * h[i + 1]
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]
        h[i] = temp
    cs_k, sn_k = givens_rotation(h[k - 1], h[k])
    h[k - 1] = cs_k * h[k - 1] + sn_k * h[k]
    h[k] = 0.0
    return h, cs_k, sn_k


def arnoldi(A, Q, k):
    """
    Arnoldi 正交化过程。
    
    生成 Krylov 子空间 K_k(A, r0) 的标准正交基 Q_k，以及
    上 Hessenberg 矩阵 H_k 满足:
      A * Q_k = Q_{k+1} * H_k
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        系统矩阵。
    Q : ndarray, shape (n, m)
        当前正交基矩阵。
    k : int
        当前 Arnoldi 步（1-based）。
    
    Returns
    -------
    h : ndarray, shape (k+1,)
        H 的第 k 列。
    q : ndarray, shape (n,)
        新的正交基向量。
    """
    q = A @ Q[:, k - 1]
    h = np.zeros(k + 1)
    for i in range(k):
        h[i] = np.dot(q, Q[:, i])
        q = q - h[i] * Q[:, i]
    h[k] = np.linalg.norm(q)
    if abs(h[k]) < 1e-14:
        q = np.zeros_like(q)
    else:
        q = q / h[k]
    return h, q


def gmres_solve(A, b, x0=None, max_iterations=100, threshold=1e-6):
    """
    使用 GMRES 算法求解线性系统 A x = b。
    
    GMRES 最小化残差范数:
      min ||A x - b||_2  over x in x0 + K_k(A, r0)
    
    算法步骤:
    1. 初始化残差 r0 = b - A*x0
    2. 构建 Krylov 子空间基 Q_k (Arnoldi)
    3. 通过 Givens 旋转将 Hessenberg 矩阵 H 化为上三角 R
    4. 求解最小二乘问题 ||beta*e1 - H*y||
    5. 更新解 x = x0 + Q_k * y
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        系数矩阵。
    b : ndarray, shape (n,)
        右端项。
    x0 : ndarray, shape (n,), optional
        初始猜测。
    max_iterations : int
        最大迭代次数。
    threshold : float
        相对残差收敛阈值。
    
    Returns
    -------
    x : ndarray, shape (n,)
        解向量。
    errors : ndarray
        每步的相对残差历史。
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    m = min(max_iterations, n)
    r = b - A @ x0
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-14:
        b_norm = 1.0
    error = np.linalg.norm(r) / b_norm
    sn = np.zeros(m)
    cs = np.zeros(m)
    e1 = np.zeros(n)
    e1[0] = 1.0
    errors = [error]
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-14:
        return x0.copy(), np.array(errors)
    Q = np.zeros((n, m + 1))
    Q[:, 0] = r / r_norm
    beta = r_norm * e1
    H = np.zeros((m + 1, m))
    for k in range(1, m + 1):
        h, q = arnoldi(A, Q, k)
        H[:k + 1, k - 1] = h
        Q[:, k] = q
        # 应用 Givens 旋转
        H[:k + 1, k - 1], cs[k - 1], sn[k - 1] = apply_givens_rotation(
            H[:k + 1, k - 1].copy(), cs, sn, k
        )
        beta[k] = -sn[k - 1] * beta[k - 1]
        beta[k - 1] = cs[k - 1] * beta[k - 1]
        error = abs(beta[k]) / b_norm
        errors.append(error)
        if error <= threshold:
            break
    # 求解上三角系统 H(1:k, 1:k) * y = beta(1:k)
    k_eff = k if error <= threshold else m
    y = np.linalg.solve(H[:k_eff, :k_eff], beta[:k_eff])
    x = x0 + Q[:, :k_eff] @ y
    return x, np.array(errors)


def build_helmholtz_matrix_1d(nx, dx, c, omega, pml_width=10, pml_sigma_max=1000.0):
    """
    构建一维 Helmholtz 方程的复对称离散矩阵。
    
    方程:
      d^2u/dx^2 + (omega/c(x))^2 * u = f(x)
    
    采用 Perfectly Matched Layer (PML) 吸收边界条件:
      s(x) = 1 + i * sigma(x) / omega
      d/dx -> 1/s(x) * d/dx
    
    PML 衰减函数:
      sigma(x) = sigma_max * (d / pml_width)^2
    其中 d 为到边界的距离。
    
    离散格式（中心差分）:
      (u_{i-1} - 2u_i + u_{i+1}) / dx^2 + k_i^2 * u_i = f_i
    
    Parameters
    ----------
    nx : int
        网格点数。
    dx : float
        网格间距。
    c : ndarray, shape (nx,)
        速度模型。
    omega : float
        角频率。
    pml_width : int
        PML 层宽度（网格点数）。
    pml_sigma_max : float
        PML 最大衰减系数。
    
    Returns
    -------
    A : ndarray, shape (nx, nx)
        Helmholtz 离散矩阵（复数）。
    """
    c = np.asarray(c, dtype=float)
    k2 = (omega / c) ** 2
    # PML 坐标拉伸
    sigma = np.zeros(nx)
    for i in range(nx):
        if i < pml_width:
            d = (pml_width - i) / pml_width
            sigma[i] = pml_sigma_max * d ** 2
        elif i >= nx - pml_width:
            d = (i - (nx - pml_width - 1)) / pml_width
            sigma[i] = pml_sigma_max * d ** 2
    s = 1.0 + 1j * sigma / omega
    # TODO: 构建 Helmholtz 方程的离散矩阵（考虑 PML 吸收边界）
    # 科学知识：一维 Helmholtz 方程 d²u/dx² + (ω/c)²·u = f 的离散化
    # PML 坐标拉伸：s(x) = 1 + i·σ(x)/ω，其中 σ 为 PML 衰减函数
    # 离散格式（中心差分，考虑 PML 拉伸）：
    #   A[i,i-1] = 1/(dx²·s_i²)
    #   A[i,i]   = -2/(dx²·s_i²) + k²_i
    #   A[i,i+1] = 1/(dx²·s_i²)
    # 要求：使用稠密矩阵返回 shape (nx, nx) 的复数矩阵 A
    pass


def solve_helmholtz_1d(nx, dx, c, omega, source_pos, source_amp=1.0,
                        max_iter=200, tol=1e-8):
    """
    求解一维 Helmholtz 方程。
    
    Parameters
    ----------
    nx : int
        网格点数。
    dx : float
        网格间距。
    c : ndarray, shape (nx,)
        速度模型。
    omega : float
        角频率。
    source_pos : int
        震源位置索引。
    source_amp : float
        震源振幅。
    max_iter : int
        GMRES 最大迭代次数。
    tol : float
        收敛阈值。
    
    Returns
    -------
    u : ndarray, shape (nx,)
        波场（复数）。
    residuals : ndarray
        GMRES 残差历史。
    """
    A = build_helmholtz_matrix_1d(nx, dx, c, omega)
    b = np.zeros(nx, dtype=complex)
    if 0 <= source_pos < nx:
        b[source_pos] = source_amp
    x0 = np.zeros(nx, dtype=complex)
    # 将复数系统转换为 2N x 2N 实数系统以使用实数 GMRES
    # [ Re(A)  -Im(A) ] [ Re(x) ] = [ Re(b) ]
    # [ Im(A)   Re(A) ] [ Im(x) ] = [ Im(b) ]
    n = nx
    A_real = np.zeros((2 * n, 2 * n))
    A_real[:n, :n] = A.real
    A_real[:n, n:] = -A.imag
    A_real[n:, :n] = A.imag
    A_real[n:, n:] = A.real
    b_real = np.zeros(2 * n)
    b_real[:n] = b.real
    b_real[n:] = b.imag
    x_sol, residuals = gmres_solve(A_real, b_real, x0=np.zeros(2 * n),
                                    max_iterations=max_iter, threshold=tol)
    u = x_sol[:n] + 1j * x_sol[n:]
    return u, residuals
