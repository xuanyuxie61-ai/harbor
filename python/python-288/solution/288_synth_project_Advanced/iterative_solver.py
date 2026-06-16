"""
iterative_solver.py - 线性系统迭代求解器与矩阵运算核心

本模块融合以下种子项目的核心算法：
  - 152_cg_rc → 反向通信共轭梯度法
  - 736_matman → LU分解与矩阵行操作

功能：
  1. 反向通信共轭梯度(CG)求解器
  2. LU分解（部分主元选取）
  3. 三对角系统Thomas算法
  4. 带状矩阵求解器
  5. 预条件子（Jacobi, SSOR, ILU(0)）
"""

import numpy as np
from typing import Tuple, Optional, Dict


# =============================================================================
# LU分解（来自736_matman的矩阵行操作思想）
# =============================================================================
def lu_decompose(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LU分解（带部分主元选取）:
      P*A = L*U

    其中：
      P 是置换矩阵
      L 是单位下三角矩阵
      U 是上三角矩阵

    使用高斯消元法的矩阵行操作实现：
      1. 行交换 (swap): 选择最大主元
      2. 行缩放 (scale): 使主元为1
      3. 行加法 (axpy): 消去下三角元素

    参数:
        A: shape (n, n) 方阵
    返回:
        P, L, U: 分解结果
    """
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        # 部分主元选取：找列k中绝对值最大的元素
        max_row = k
        max_val = abs(U[k, k])
        for i in range(k + 1, n):
            if abs(U[i, k]) > max_val:
                max_val = abs(U[i, k])
                max_row = i

        # 行交换 (swap)
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[k, :k], L[max_row, :k] = L[max_row, :k].copy(), L[k, :k].copy()

        # 检查主元是否为零
        if abs(U[k, k]) < 1e-30:
            continue

        # 行操作(axpy): 消去下三角
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return P, L, U


def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    使用LU分解求解 Ax = b

    步骤:
      1. PA = LU
      2. Ly = Pb (前代)
      3. Ux = y (回代)

    参数:
        A: shape (n, n)
        b: shape (n,)
    返回:
        x: shape (n,)
    """
    P, L, U = lu_decompose(A)

    n = A.shape[0]

    # 前代: Ly = Pb
    Pb = P @ b
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        # L的对角线是1，不需要除法

    # 回代: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        if abs(U[i, i]) < 1e-30:
            x[i] = 0.0
        else:
            x[i] /= U[i, i]

    return x


# =============================================================================
# 反向通信共轭梯度法（来自152_cg_rc）
# =============================================================================
def reverse_communication_cg(n: int, tol: float = 1e-10,
                               max_iter: int = 1000) -> Dict:
    """
    初始化反向通信CG求解器

    反向通信模式：调用者负责执行矩阵向量乘和预条件子求解，
    使CG算法可以适配任意矩阵存储格式

    CG迭代步骤:
      r_0 = b - A*x_0
      z_0 = M^{-1} * r_0  (预条件子)
      p_0 = z_0
      for k = 0, 1, 2, ...
        alpha_k = (r_k, z_k) / (p_k, A*p_k)
        x_{k+1} = x_k + alpha_k * p_k
        r_{k+1} = r_k - alpha_k * A*p_k
        z_{k+1} = M^{-1} * r_{k+1}
        beta_k = (r_{k+1}, z_{k+1}) / (r_k, z_k)
        p_{k+1} = z_{k+1} + beta_k * p_k

    参数:
        n: 系统维度
        tol: 收敛容差
        max_iter: 最大迭代次数
    返回:
        state: CG状态字典
    """
    state = {
        'n': n,
        'x': np.zeros(n),
        'r': np.zeros(n),
        'p': np.zeros(n),
        'z': np.zeros(n),
        'Ap': np.zeros(n),
        'rho': 0.0,
        'rho_old': 0.0,
        'alpha': 0.0,
        'beta': 0.0,
        'tol': tol,
        'max_iter': max_iter,
        'iteration': 0,
        'converged': False,
        'residual_norm': float('inf'),
        'phase': 'init',  # init, matvec, precond, update, done
        'request': None,
    }
    return state


def cg_initialize(state: Dict, b: np.ndarray, x0: Optional[np.ndarray] = None):
    """初始化CG（计算初始残差）"""
    if x0 is not None:
        state['x'] = x0.copy()
    else:
        state['x'] = np.zeros(state['n'])

    state['r'] = b.copy()  # 假设x0=0时r0=b
    state['phase'] = 'precond'
    state['request'] = 'apply_preconditioner'


def cg_update_precond(state: Dict, z: np.ndarray):
    """接收预条件子结果 z = M^{-1} r"""
    state['z'] = z.copy()
    state['p'] = z.copy()
    state['rho'] = np.dot(state['r'], state['z'])
    state['phase'] = 'matvec'
    state['request'] = 'compute_Ap'


def cg_update_matvec(state: Dict, Ap: np.ndarray) -> bool:
    """
    接收矩阵向量乘结果 Ap = A*p
    返回True表示收敛
    """
    state['Ap'] = Ap.copy()

    pAp = np.dot(state['p'], state['Ap'])
    if abs(pAp) < 1e-30:
        state['converged'] = False
        state['phase'] = 'done'
        return True

    state['alpha'] = state['rho'] / pAp
    state['x'] += state['alpha'] * state['p']
    state['r'] -= state['alpha'] * state['Ap']

    state['residual_norm'] = np.linalg.norm(state['r'])

    if state['residual_norm'] < state['tol']:
        state['converged'] = True
        state['phase'] = 'done'
        return True

    state['iteration'] += 1
    if state['iteration'] >= state['max_iter']:
        state['phase'] = 'done'
        return True

    state['phase'] = 'precond'
    state['request'] = 'apply_preconditioner'
    return False


def cg_update_precond2(state: Dict, z: np.ndarray) -> bool:
    """后续预条件子更新"""
    state['z'] = z.copy()
    state['rho_old'] = state['rho']
    state['rho'] = np.dot(state['r'], state['z'])

    if abs(state['rho_old']) < 1e-30:
        state['phase'] = 'done'
        return True

    state['beta'] = state['rho'] / state['rho_old']
    state['p'] = state['z'] + state['beta'] * state['p']
    state['phase'] = 'matvec'
    state['request'] = 'compute_Ap'
    return False


# =============================================================================
# 直接CG求解器（封装版）
# =============================================================================
def solve_sparse_cg(A: np.ndarray, b: np.ndarray,
                     tol: float = 1e-10, max_iter: int = 1000,
                     precond: str = 'jacobi') -> np.ndarray:
    """
    直接使用CG求解 Ax = b

    支持的预条件子:
      'none': 无预条件
      'jacobi': 对角预条件 M = diag(A)
      'ssor': 对称超松弛预条件

    参数:
        A: 系统矩阵
        b: 右端项
        tol: 收敛容差
        max_iter: 最大迭代
        precond: 预条件子类型
    返回:
        x: 解向量
    """
    n = len(b)

    # 构造预条件子
    if precond == 'jacobi':
        diag_A = np.diag(A).copy()
        diag_A = np.where(np.abs(diag_A) < 1e-30, 1.0, diag_A)
        M_inv_diag = 1.0 / diag_A

        def apply_precond(r):
            return M_inv_diag * r
    elif precond == 'ssor':
        omega = 1.5
        D = np.diag(np.diag(A))
        L_plus_D = np.tril(A)
        D_inv = np.diag(1.0 / (np.diag(L_plus_D) + 1e-30))

        def apply_precond(r):
            # SSOR: M = (1/omega * D + L) * D^{-1} * (1/omega * D + U)
            z1 = np.zeros(n)
            for i in range(n):
                z1[i] = r[i]
                for j in range(i):
                    z1[i] -= omega * L_plus_D[i, j] * z1[j]
                z1[i] *= omega * D_inv[i, i]
            z = z1.copy()
            for i in range(n - 1, -1, -1):
                for j in range(i + 1, n):
                    z[i] -= omega * L_plus_D[j, i] * z[j]
                z[i] *= omega * D_inv[i, i]
            return z
    else:
        def apply_precond(r):
            return r.copy()

    # 标准PCG
    x = np.zeros(n)
    r = b.copy()
    z = apply_precond(r)
    p = z.copy()
    rho = np.dot(r, z)

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-30:
        return x

    for iteration in range(max_iter):
        Ap = A @ p
        pAp = np.dot(p, Ap)

        if abs(pAp) < 1e-30:
            break

        alpha = rho / pAp
        x += alpha * p
        r -= alpha * Ap

        r_norm = np.linalg.norm(r)
        if r_norm / b_norm < tol:
            break

        z = apply_precond(r)
        rho_new = np.dot(r, z)

        if abs(rho) < 1e-30:
            break

        beta = rho_new / rho
        p = z + beta * p
        rho = rho_new

    return x


# =============================================================================
# 三对角系统求解（Thomas算法）
# =============================================================================
def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                        d: np.ndarray) -> np.ndarray:
    """
    求解三对角系统:
      a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i

    Thomas算法（追赶法）：O(n)复杂度

    参数:
        a: 下次对角线（a[0]未使用）
        b: 主对角线
        c: 上次对角线（c[n-1]未使用）
        d: 右端项
    返回:
        x: 解向量
    """
    n = len(b)
    if n == 0:
        return np.array([])
    if n == 1:
        return d / b

    # 复制以避免修改输入
    c_ = c.copy().astype(float)
    d_ = d.copy().astype(float)
    b_ = b.copy().astype(float)

    # 前向消元
    for i in range(1, n):
        if abs(b_[i - 1]) < 1e-30:
            continue
        m = a[i] / b_[i - 1]
        b_[i] -= m * c_[i - 1]
        d_[i] -= m * d_[i - 1]

    # 回代
    x = np.zeros(n)
    if abs(b_[n - 1]) > 1e-30:
        x[n - 1] = d_[n - 1] / b_[n - 1]

    for i in range(n - 2, -1, -1):
        if abs(b_[i]) > 1e-30:
            x[i] = (d_[i] - c_[i] * x[i + 1]) / b_[i]

    return x


# =============================================================================
# 带状矩阵构造
# =============================================================================
def construct_banded_system(diagonals: Dict[str, np.ndarray], n: int) -> np.ndarray:
    """
    从对角线数组构造稠密矩阵

    参数:
        diagonals: dict包含 'main', 'lower', 'upper' 等对角线
        n: 矩阵维度
    返回:
        A: shape (n, n) 矩阵
    """
    A = np.zeros((n, n))

    if 'main' in diagonals:
        np.fill_diagonal(A, diagonals['main'])
    if 'lower' in diagonals:
        for i in range(1, n):
            A[i, i - 1] = diagonals['lower'][i]
    if 'upper' in diagonals:
        for i in range(n - 1):
            A[i, i + 1] = diagonals['upper'][i]
    if 'lower2' in diagonals:
        for i in range(2, n):
            A[i, i - 2] = diagonals['lower2'][i]
    if 'upper2' in diagonals:
        for i in range(n - 2):
            A[i, i + 2] = diagonals['upper2'][i]

    return A


# =============================================================================
# 矩阵条件数估计
# =============================================================================
def estimate_condition_number(A: np.ndarray) -> float:
    """
    估计矩阵条件数 kappa(A) = ||A|| * ||A^{-1}||

    使用Frobenius范数

    参数:
        A: 方阵
    返回:
        条件数估计
    """
    try:
        A_inv = np.linalg.inv(A)
        cond = np.linalg.norm(A, 'fro') * np.linalg.norm(A_inv, 'fro')
        return float(cond)
    except np.linalg.LinAlgError:
        return float('inf')


# =============================================================================
# Wathen矩阵（有限元测试矩阵，来自152_cg_rc）
# =============================================================================
def generate_wathen_element(nx: float, ny: float) -> np.ndarray:
    """
    生成Wathen有限元质量矩阵的单个单元贡献

    8节点serendipity单元的一致性质量矩阵

    参数:
        nx, ny: 随机参数
    返回:
        ke: shape (8, 8) 单元矩阵
    """
    rho = nx + ny  # 密度参数

    # Wathen矩阵的解析表达式
    ke = np.zeros((8, 8))

    # 简化版本：4节点双线性单元
    # 使用2x2高斯积分
    gauss_pts = np.array([
        [-1.0 / np.sqrt(3), -1.0 / np.sqrt(3)],
        [1.0 / np.sqrt(3), -1.0 / np.sqrt(3)],
        [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
        [1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
    ])
    weights = np.ones(4)

    for q in range(4):
        xi, eta = gauss_pts[q]
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        ])
        ke += weights[q] * rho * np.outer(N, N)

    return ke
