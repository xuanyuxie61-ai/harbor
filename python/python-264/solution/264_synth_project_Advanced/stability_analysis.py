# -*- coding: utf-8 -*-
"""
stability_analysis.py
=====================

数值稳定性分析模块.

本模块实现磁层粒子输运方程离散化后的稳定性分析:
  - von Neumann 稳定性分析 (傅里叶模式分析)
  - CFL 条件计算
  - 矩阵稳定性分析 (谱半径)
  - Gram-Schmidt 正交化 (特征模式分解)
  - 矩阵指数时间推进

物理背景:

对于半离散化的 Fokker-Planck 方程:
  df/dt = A * f + S

其中 A 为离散算子矩阵, S 为源项.

稳定性条件:
  1. von Neumann 条件: 所有傅里叶模式的放大因子 |g(k)| <= 1
  2. CFL 条件: dt <= C * dx / max(|v|)  (对流)
              dt <= C * dx^2 / max(D)  (扩散)
  3. 矩阵稳定性: dt * max(Re(lambda_i)) <= 0  (A 的特征值)

对于高阶有限差分格式, von Neumann 分析给出:
  2阶中心差分: g(k) = 1 - 4*r*sin^2(k*dx/2)  (扩散方程)
  4阶中心差分: g(k) = 1 - r*(4/3)*sin^2(k*dx) + r*(1/3)*sin^2(k*dx/2)
  WENO5: g(k) 由非线性权重决定, 需数值分析

参考文献:
  [1] Strikwerda, J.C., "Finite Difference Schemes and PDEs", SIAM (2004)
  [2] LeVeque, R.J., "Finite Difference Methods for ODEs and PDEs", Cambridge (2007)
  [3] Golub, G.H. & Van Loan, C.F., "Matrix Computations", JHU (2013)
"""

import numpy as np
import physical_constants as pc
import high_order_fd as hofd


# =============================================================================
#  von Neumann 稳定性分析
# =============================================================================

def von_neumann_analysis_diffusion(scheme='central_2nd', r_values=None):
    """
    扩散方程的 von Neumann 稳定性分析.

    测试方程: du/dt = D * d^2u/dx^2
    离散化: u_j^{n+1} = g(k) * u_j^n

    对于 2 阶中心差分:
      u_j^{n+1} = u_j^n + r * (u_{j+1}^n - 2*u_j^n + u_{j-1}^n)
      g(k) = 1 + r * (exp(ik*dx) - 2 + exp(-ik*dx))
           = 1 - 4*r*sin^2(k*dx/2)

    稳定性条件: |g(k)| <= 1  =>  r <= 1/2

    对于 4 阶中心差分:
      u_j^{n+1} = u_j^n + r * (-1/12*u_{j+2} + 4/3*u_{j+1} - 5/2*u_j + 4/3*u_{j-1} - 1/12*u_{j-2})
      g(k) = 1 + r * (-1/6*cos(2*k*dx) + 8/3*cos(k*dx) - 5/2)

    参数
    ----
    scheme : str
        差分格式: 'central_2nd', 'central_4th'
    r_values : ndarray, optional
        r = D*dt/dx^2 的值数组

    返回
    -------
    results : dict
        稳定性分析结果:
          - r_max: 最大允许 r 值
          - g_curves: 放大因子曲线
          - stable: 是否稳定
    """
    if r_values is None:
        r_values = np.linspace(0.0, 1.0, 101)

    # 波数范围
    theta = np.linspace(0, np.pi, 200)  # theta = k*dx

    results = {'scheme': scheme, 'r_values': r_values}

    if scheme == 'central_2nd':
        # g(k) = 1 - 4*r*sin^2(theta/2)
        g = np.zeros((len(r_values), len(theta)))
        for i, r in enumerate(r_values):
            g[i, :] = 1.0 - 4.0 * r * np.sin(theta / 2.0)**2
        results['r_max'] = 0.5
        results['g'] = g

    elif scheme == 'central_4th':
        # g(k) = 1 + r * (-1/6*cos(2*theta) + 8/3*cos(theta) - 5/2)
        g = np.zeros((len(r_values), len(theta)))
        for i, r in enumerate(r_values):
            g[i, :] = 1.0 + r * (-1.0/6.0 * np.cos(2*theta) + 8.0/3.0 * np.cos(theta) - 5.0/2.0)
        # 稳定性条件: r <= 6/7 ~ 0.857
        results['r_max'] = 6.0 / 7.0
        results['g'] = g

    else:
        raise ValueError(f"未知格式: {scheme}")

    # 检查稳定性
    g_max = np.max(np.abs(g), axis=1)
    results['stable'] = g_max <= 1.0 + 1e-10
    results['g_max'] = g_max

    return results


def von_neumann_analysis_advection(scheme='upwind_1st', cfl_values=None):
    """
    对流方程的 von Neumann 稳定性分析.

    测试方程: du/dt + a * du/dx = 0
    离散化: u_j^{n+1} = g(k) * u_j^n

    对于 1 阶迎风:
      u_j^{n+1} = u_j^n - nu * (u_j^n - u_{j-1}^n)
      g(k) = 1 - nu * (1 - exp(-ik*dx))
           = 1 - nu + nu*cos(theta) - i*nu*sin(theta)
      |g|^2 = (1 - nu + nu*cos(theta))^2 + (nu*sin(theta))^2
            = 1 - 2*nu*(1-nu)*(1 - cos(theta))

    稳定性条件: nu <= 1 (CFL 条件)

    参数
    ----
    scheme : str
        差分格式: 'upwind_1st', 'lax_friedrichs', 'leapfrog'
    cfl_values : ndarray, optional
        CFL 数 nu = a*dt/dx 的值数组

    返回
    -------
    results : dict
        稳定性分析结果
    """
    if cfl_values is None:
        cfl_values = np.linspace(0.0, 2.0, 201)

    theta = np.linspace(0, np.pi, 200)
    results = {'scheme': scheme, 'cfl_values': cfl_values}

    if scheme == 'upwind_1st':
        # |g|^2 = 1 - 2*nu*(1-nu)*(1 - cos(theta))
        g_sq = np.zeros((len(cfl_values), len(theta)))
        for i, nu in enumerate(cfl_values):
            g_sq[i, :] = 1.0 - 2.0 * nu * (1.0 - nu) * (1.0 - np.cos(theta))
        results['cfl_max'] = 1.0
        results['g_squared'] = g_sq

    elif scheme == 'lax_friedrichs':
        # |g|^2 = 1 - (1 - nu^2) * sin^2(theta)
        g_sq = np.zeros((len(cfl_values), len(theta)))
        for i, nu in enumerate(cfl_values):
            g_sq[i, :] = 1.0 - (1.0 - nu**2) * np.sin(theta)**2
        results['cfl_max'] = 1.0
        results['g_squared'] = g_sq

    elif scheme == 'leapfrog':
        # Leapfrog 是三层格式, 需要更复杂的分析
        # 简化: 检查 |g|^2 <= 1 + epsilon
        results['cfl_max'] = 1.0
        results['g_squared'] = None

    else:
        raise ValueError(f"未知格式: {scheme}")

    # 检查稳定性
    if results['g_squared'] is not None:
        g_max = np.sqrt(np.maximum(results['g_squared'], 0.0))
        results['stable'] = np.max(g_max, axis=1) <= 1.0 + 1e-10
        results['g_max'] = np.max(g_max, axis=1)
    else:
        results['stable'] = cfl_values <= results['cfl_max']
        results['g_max'] = None

    return results


# =============================================================================
#  CFL 条件计算
# =============================================================================

def compute_cfl_timestep(grid, D_LL, v_drift=None, cfl_number=None):
    """
    计算满足 CFL 条件的最大时间步长.

    对于扩散方程: dt <= C * dL^2 / D_LL_max
    对于对流方程: dt <= C * dL / v_max
    对于对流-扩散: dt <= min(dt_diff, dt_adv)

    参数
    ----
    grid : MagnetosphereGrid
        相空间网格
    D_LL : ndarray
        径向扩散系数
    v_drift : ndarray, optional
        漂移速度 (若存在对流项)
    cfl_number : float, optional
        CFL 数, 默认 pc.CFL_DEFAULT

    返回
    -------
    dt_cfl : float
        满足 CFL 条件的最大时间步长
    dt_diff : float
        扩散限制的时间步长
    dt_adv : float
        对流限制的时间步长
    """
    cfl = pc.CFL_DEFAULT if cfl_number is None else cfl_number

    dL = grid.dL
    D_max = np.max(np.abs(D_LL))

    # 扩散 CFL: dt <= C * dL^2 / D_max
    if D_max > pc.EPSILON_NUM:
        dt_diff = cfl * dL**2 / D_max
    else:
        dt_diff = np.inf

    # 对流 CFL: dt <= C * dL / v_max
    if v_drift is not None:
        v_max = np.max(np.abs(v_drift))
        if v_max > pc.EPSILON_NUM:
            dt_adv = cfl * dL / v_max
        else:
            dt_adv = np.inf
    else:
        dt_adv = np.inf

    dt_cfl = min(dt_diff, dt_adv)

    return dt_cfl, dt_diff, dt_adv


# =============================================================================
#  矩阵稳定性分析
# =============================================================================

def build_diffusion_matrix(grid, D_LL):
    """
    构建径向扩散算子矩阵.

    离散化:
      d/dL (D_LL * df/dL) ~ [D_{i+1/2}*(f_{i+1}-f_i) - D_{i-1/2}*(f_i-f_{i-1})] / dL^2

    参数
    ----
    grid : MagnetosphereGrid
        相空间网格
    D_LL : ndarray
        扩散系数 (在网格点上)

    返回
    -------
    A : ndarray, shape (n_L, n_L)
        离散扩散算子矩阵
    """
    n_L = grid.n_L
    dL = grid.dL
    A = np.zeros((n_L, n_L))

    # 半节点扩散系数 (算术平均)
    D_half = 0.5 * (D_LL[:-1] + D_LL[1:])

    # 内部节点
    for i in range(1, n_L - 1):
        D_plus = D_half[i] if i < len(D_half) else D_LL[-1]
        D_minus = D_half[i-1] if i-1 >= 0 else D_LL[0]

        A[i, i-1] = D_minus / dL**2
        A[i, i] = -(D_plus + D_minus) / dL**2
        A[i, i+1] = D_plus / dL**2

    # 边界: Dirichlet (f=0)
    A[0, 0] = -1.0 / dL**2
    A[0, 1] = 1.0 / dL**2
    A[-1, -1] = -1.0 / dL**2
    A[-1, -2] = 1.0 / dL**2

    return A


def matrix_stability_analysis(A, dt=None):
    """
    矩阵稳定性分析.

    对于半离散系统 df/dt = A*f, 时间推进稳定性要求:
      rho(I + dt*A) <= 1  (显式 Euler)
      或 dt * max(Re(lambda_i)) <= 0

    参数
    ----
    A : ndarray, shape (N, N)
        离散算子矩阵
    dt : float, optional
        时间步长

    返回
    -------
    results : dict
        稳定性分析结果:
          - eigenvalues: 特征值
          - spectral_radius: 谱半径
          - max_real_part: 最大实部
          - dt_max: 最大允许时间步长 (显式 Euler)
          - stable: 是否稳定
    """
    # 计算特征值
    eigenvalues = np.linalg.eigvals(A)

    # 谱半径
    spectral_radius = np.max(np.abs(eigenvalues))

    # 最大实部
    max_real = np.max(np.real(eigenvalues))

    # 显式 Euler 稳定性: |1 + dt*lambda| <= 1
    # 对于实负特征值: dt <= 2 / |lambda_max|
    if max_real < -pc.EPSILON_NUM:
        dt_max_euler = 2.0 / np.abs(max_real)
    else:
        dt_max_euler = np.inf

    results = {
        'eigenvalues': eigenvalues,
        'spectral_radius': float(spectral_radius),
        'max_real_part': float(max_real),
        'min_real_part': float(np.min(np.real(eigenvalues))),
        'dt_max_euler': float(dt_max_euler),
    }

    # 检查给定 dt 的稳定性
    if dt is not None:
        amplification = np.abs(1.0 + dt * eigenvalues)
        results['max_amplification'] = float(np.max(amplification))
        results['stable'] = results['max_amplification'] <= 1.0 + 1e-10
    else:
        results['stable'] = max_real <= pc.EPSILON_NUM

    return results


# =============================================================================
#  Gram-Schmidt 正交化
# =============================================================================

def modified_gram_schmidt(V, tolerance=1e-12):
    """
    改进的 Gram-Schmidt 正交化.

    算法 (MGS):
      对于 k = 1, 2, ..., n:
        v_k = v_k - sum_{j=1}^{k-1} (v_j^T * v_k) * v_j
        v_k = v_k / ||v_k||

    MGS 比经典 GS 数值稳定性更好, 正交性误差为 O(epsilon)
    而非 O(epsilon * kappa(A)).

    物理应用:
      在磁层辐射带模拟中, 用于:
        - 扩散算子的特征模式正交化
        - Arnoldi 迭代中的 Krylov 子空间基
        - 投掷角分布的球谐函数展开

    参数
    ----
    V : ndarray, shape (m, n)
        输入向量矩阵 (列向量)
    tolerance : float
        线性相关检测容差

    返回
    -------
    Q : ndarray, shape (m, n)
        正交化后的矩阵
    R : ndarray, shape (n, n)
        上三角矩阵, 满足 V = Q*R
    rank : int
        数值秩
    """
    m, n = V.shape
    Q = V.copy().astype(np.float64)
    R = np.zeros((n, n))
    rank = 0

    for j in range(n):
        # 减去已正交化的分量
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], Q[:, j])
            Q[:, j] -= R[i, j] * Q[:, i]

        # 归一化
        norm = np.linalg.norm(Q[:, j])
        if norm > tolerance:
            R[j, j] = norm
            Q[:, j] /= norm
            rank += 1
        else:
            # 线性相关, 设为零
            R[j, j] = 0.0
            Q[:, j] = 0.0

    return Q, R, rank


def classical_gram_schmidt(V, tolerance=1e-12):
    """
    经典 Gram-Schmidt 正交化.

    算法 (CGS):
      对于 k = 1, 2, ..., n:
        v_k = v_k - sum_{j=1}^{k-1} (v_j^T * v_k) * v_j  (一次性计算所有投影)
        v_k = v_k / ||v_k||

    注意: CGS 的数值稳定性较差, 推荐使用 MGS.

    参数
    ----
    V : ndarray, shape (m, n)
        输入向量矩阵
    tolerance : float
        线性相关检测容差

    返回
    -------
    Q : ndarray
        正交化后的矩阵
    R : ndarray
        上三角矩阵
    rank : int
        数值秩
    """
    m, n = V.shape
    Q = np.zeros_like(V, dtype=np.float64)
    R = np.zeros((n, n))
    rank = 0

    for j in range(n):
        v = V[:, j].copy()
        # 一次性计算所有投影
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            v -= R[i, j] * Q[:, i]

        norm = np.linalg.norm(v)
        if norm > tolerance:
            R[j, j] = norm
            Q[:, j] = v / norm
            rank += 1
        else:
            R[j, j] = 0.0

    return Q, R, rank


def arnoldi_iteration(A, v0, max_iter=20, tolerance=1e-10):
    """
    Arnoldi 迭代构造 Krylov 子空间.

    算法:
      输入: 矩阵 A, 初始向量 v0
      初始化: q1 = v0 / ||v0||
      对于 j = 1, 2, ..., m:
        w = A * q_j
        对于 i = 1, ..., j:
          h_{ij} = q_i^T * w
          w = w - h_{ij} * q_i
        h_{j+1,j} = ||w||
        q_{j+1} = w / h_{j+1,j}

    物理应用:
      用于计算大稀疏矩阵的特征值 (辐射带扩散算子).

    参数
    ----
    A : ndarray 或 callable
        矩阵或矩阵-向量乘积函数
    v0 : ndarray
        初始向量
    max_iter : int
        最大迭代次数
    tolerance : float
        收敛容差

    返回
    -------
    Q : ndarray, shape (n, m)
        正交基矩阵
    H : ndarray, shape (m+1, m)
        海森堡矩阵
    eigenvalues : ndarray
        Ritz 值 (A 的近似特征值)
    """
    n = len(v0)
    Q = np.zeros((n, max_iter + 1))
    H = np.zeros((max_iter + 1, max_iter))

    # 初始化
    beta = np.linalg.norm(v0)
    if beta < tolerance:
        return Q, H, np.array([])
    Q[:, 0] = v0 / beta

    # Arnoldi 迭代
    m_actual = 0
    for j in range(max_iter):
        # 矩阵-向量乘积
        if callable(A):
            w = A(Q[:, j])
        else:
            w = A @ Q[:, j]

        # MGS 正交化
        for i in range(j + 1):
            H[i, j] = np.dot(Q[:, i], w)
            w -= H[i, j] * Q[:, i]

        # 重正交化 (两次 MGS 提高稳定性)
        for i in range(j + 1):
            s = np.dot(Q[:, i], w)
            H[i, j] += s
            w -= s * Q[:, i]

        H[j+1, j] = np.linalg.norm(w)
        m_actual = j + 1

        if H[j+1, j] < tolerance:
            break

        Q[:, j+1] = w / H[j+1, j]

    # 计算 H 的特征值 (Ritz 值)
    H_m = H[:m_actual, :m_actual]
    eigenvalues = np.linalg.eigvals(H_m)

    return Q[:, :m_actual+1], H[:m_actual+1, :m_actual], eigenvalues


# =============================================================================
#  矩阵指数 (用于精确时间推进)
# =============================================================================

def matrix_exponential(A, dt, method='pade'):
    """
    计算矩阵指数 exp(A*dt).

    方法:
      - 'pade': Padé 近似 + 缩放-平方
      - 'eigen': 特征值分解
      - 'taylor': Taylor 级数

    物理应用:
      精确求解 df/dt = A*f:
        f(t+dt) = exp(A*dt) * f(t)

    参数
    ----
    A : ndarray
        矩阵
    dt : float
        时间步长
    method : str
        计算方法

    返回
    -------
    expA : ndarray
        矩阵指数
    """
    if method == 'pade':
        # 使用 scipy 的 Padé 近似 (简化版)
        # 缩放-平方: exp(A) = (exp(A/2^s))^{2^s}
        norm_A = np.linalg.norm(A * dt, ord=np.inf)
        s = max(0, int(np.ceil(np.log2(norm_A + 1e-16))))
        A_scaled = A * dt / (2**s)

        # 6 阶 Padé 近似
        I = np.eye(len(A))
        A2 = A_scaled @ A_scaled
        A4 = A2 @ A2
        A6 = A4 @ A2

        # Padé 系数
        b = [1.0, 0.5, 1.0/9.0, 1.0/72.0, 1.0/1008.0]
        U = A_scaled @ (b[1]*I + b[3]*A2 + b[5]*A4 if len(b) > 4 else b[1]*I + b[3]*A2)
        V = b[0]*I + b[2]*A2 + b[4]*A4 if len(b) > 4 else b[0]*I + b[2]*A2

        # 简化: 使用 3 阶 Taylor
        expA_scaled = I + A_scaled + 0.5*A2 + A2@A_scaled/6.0

        # 平方
        expA = expA_scaled
        for _ in range(s):
            expA = expA @ expA

        return expA

    elif method == 'eigen':
        eigenvalues, eigenvectors = np.linalg.eig(A)
        exp_eigenvalues = np.exp(eigenvalues * dt)
        return eigenvectors @ np.diag(exp_eigenvalues) @ np.linalg.inv(eigenvectors)

    elif method == 'taylor':
        I = np.eye(len(A))
        expA = I.copy()
        term = I.copy()
        for k in range(1, 20):
            term = term @ (A * dt) / k
            expA = expA + term
            if np.linalg.norm(term) < 1e-15:
                break
        return expA

    else:
        raise ValueError(f"未知方法: {method}")


# =============================================================================
#  综合稳定性诊断
# =============================================================================

def comprehensive_stability_check(grid, D_LL, dt, scheme='central_2nd'):
    """
    综合稳定性检查.

    检查项目:
      1. CFL 条件
      2. von Neumann 稳定性
      3. 矩阵谱半径
      4. 显式 Euler 稳定性

    参数
    ----
    grid : MagnetosphereGrid
        相空间网格
    D_LL : ndarray
        径向扩散系数
    dt : float
        时间步长
    scheme : str
        差分格式

    返回
    -------
    report : dict
        稳定性报告
    """
    report = {}

    # 1. CFL 条件
    dt_cfl, dt_diff, dt_adv = compute_cfl_timestep(grid, D_LL)
    report['cfl'] = {
        'dt_cfl': dt_cfl,
        'dt_diff': dt_diff,
        'dt_adv': dt_adv,
        'cfl_number': dt / dt_cfl if dt_cfl < np.inf else 0.0,
        'satisfied': dt <= dt_cfl,
    }

    # 2. von Neumann 分析
    if scheme == 'central_2nd':
        vn = von_neumann_analysis_diffusion('central_2nd')
    elif scheme == 'central_4th':
        vn = von_neumann_analysis_diffusion('central_4th')
    else:
        vn = {'r_max': 0.5}
    D_max = np.max(np.abs(D_LL))
    r_actual = D_max * dt / grid.dL**2 if D_max > 0 else 0.0
    report['von_neumann'] = {
        'scheme': scheme,
        'r_actual': r_actual,
        'r_max': vn.get('r_max', 0.5),
        'satisfied': r_actual <= vn.get('r_max', 0.5),
    }

    # 3. 矩阵稳定性
    A = build_diffusion_matrix(grid, D_LL)
    mat_stab = matrix_stability_analysis(A, dt)
    report['matrix'] = mat_stab

    # 4. 综合判断
    report['overall_stable'] = (
        report['cfl']['satisfied'] and
        report['von_neumann']['satisfied'] and
        mat_stab.get('stable', True)
    )

    return report


def print_stability_report(report):
    """打印稳定性报告."""
    print("=" * 60)
    print("稳定性分析报告")
    print("=" * 60)

    print("\n[1] CFL 条件:")
    print(f"  dt_cfl = {report['cfl']['dt_cfl']:.4e} s")
    print(f"  dt_diff = {report['cfl']['dt_diff']:.4e} s")
    print(f"  当前 dt = {report['cfl']['dt_cfl'] * report['cfl']['cfl_number']:.4e} s")
    print(f"  CFL 数 = {report['cfl']['cfl_number']:.4f}")
    print(f"  满足: {report['cfl']['satisfied']}")

    print("\n[2] von Neumann 稳定性:")
    print(f"  格式: {report['von_neumann']['scheme']}")
    print(f"  r = {report['von_neumann']['r_actual']:.4f}")
    print(f"  r_max = {report['von_neumann']['r_max']:.4f}")
    print(f"  满足: {report['von_neumann']['satisfied']}")

    print("\n[3] 矩阵稳定性:")
    print(f"  谱半径 = {report['matrix']['spectral_radius']:.4e}")
    print(f"  最大实部 = {report['matrix']['max_real_part']:.4e}")
    print(f"  dt_max (Euler) = {report['matrix']['dt_max_euler']:.4e}")
    print(f"  满足: {report['matrix'].get('stable', True)}")

    print(f"\n[总结] 整体稳定: {report['overall_stable']}")


if __name__ == "__main__":
    # 简单测试
    print("--- von Neumann 扩散分析 ---")
    vn = von_neumann_analysis_diffusion('central_2nd')
    print(f"2 阶中心差分: r_max = {vn['r_max']}")

    vn4 = von_neumann_analysis_diffusion('central_4th')
    print(f"4 阶中心差分: r_max = {vn4['r_max']:.4f}")

    print("\n--- 矩阵稳定性 ---")
    A = np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]])
    stab = matrix_stability_analysis(A, dt=0.1)
    print(f"特征值: {stab['eigenvalues']}")
    print(f"稳定: {stab['stable']}")
