"""
l1_ista.py — L1 正则化近端梯度求解器模块
===========================================
来源项目映射:
  - 1115_sandyherdo_inerOsci → 惯性振子 (heavy-ball 动力学)
  - 1374_unstable_ode        → 不稳定 ODE 的稳定化积分

科学背景:
  LASSO / Basis Pursuit Denoising 问题:
      min_x  0.5 ||Ax - y||_2²  +  λ ||x||_1
  等价于近端梯度形式:
      x^{k+1} = S_{λ/L}(x^k - (1/L) A^T(Ax^k - y))
  其中 S_κ 为软阈值算子: S_κ(x) = sign(x) max(|x|-κ, 0).

  FISTA (Beck-Teboulle 2009) 通过 Nesterov 动量加速:
      y^k = x^k + ((t_k - 1)/t_{k+1})(x^k - x^{k-1})
      x^{k+1} = S_{λ/L}(y^k - (1/L) ∇f(y^k))

  惯性振子视角 (源自 1115):
      m ẍ + γ ẋ + ∇f(x) + λ ∂||x||_1 ∋ 0
  离散化得到 heavy-ball 近端方法:
      x^{k+1} = S(x^k + β(x^k - x^{k-1}) - α ∇f(x^k))
  参数选择:
      α < 2/(L + m·ω²),  β < 1 - √(αL)

  不稳定 ODE 正则化 (源自 1374):
      当 L 极大时, 显式梯度步可能发散.
      采用隐式中点法则:
        x^{k+1/2} = x^k + 0.5·h·g(x^{k+1/2})
      用 Newton 迭代求解中间步.

核心公式:
  软阈值:
    S_κ(x) = sign(x) · max(|x| - κ, 0)
  Lipschitz 常数:
    L = σ_max(A)^2 = λ_max(A^T A)
  FISTA 收敛率:
    F(x^k) - F(x*) ≤ 2L ||x^0 - x*||² / (k+1)²
"""
import numpy as np


# ----------------------------------------------------------------------
# 基本算子
# ----------------------------------------------------------------------
def soft_threshold(x, kappa):
    """软阈值算子 (proximal of L1 norm).

    S_κ(x) = sign(x) · max(|x| - κ, 0)
    次梯度形式: x ∈ argmin_z { 0.5||z - v||² + κ||z||_1 }
    """
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)


def lipschitz_constant(A, n_iter=50, tol=1e-8):
    """幂迭代估计 L = ||A||_2² = λ_max(A^T A)."""
    m, n = A.shape
    rng = np.random.RandomState(0)
    x = rng.randn(n)
    x /= np.linalg.norm(x)
    sigma = 0.0
    for _ in range(n_iter):
        y = A @ x
        z = A.T @ y
        sigma_new = np.linalg.norm(z)
        if sigma_new < 1e-14:
            return 1e-14
        x = z / sigma_new
        if abs(sigma_new - sigma) < tol * sigma_new:
            break
        sigma = sigma_new
    return float(sigma_new)


# ----------------------------------------------------------------------
# ISTA / FISTA (源自 1115 惯性振子视角)
# ----------------------------------------------------------------------
def ista_l1(A, y, lam, x0=None, max_iter=2000, tol=1e-8, adaptive_step=False):
    """ISTA 求解 LASSO.

    输入:
        A       : (m, n)
        y       : (m,)
        lam     : λ > 0
        x0      : 初始点
        adaptive_step : 是否采用回溯线搜索
    返回:
        x       : 最优解
        history : dict (obj, sparsity, residual_norm)
    """
    m, n = A.shape
    if x0 is None:
        x0 = np.zeros(n)
    x = x0.copy()
    L = lipschitz_constant(A) if not adaptive_step else 1.0 / np.linalg.norm(A, 'fro') ** 2
    step = 1.0 / L

    AtA = A.T @ A
    Aty = A.T @ y
    obj_hist, sp_hist = [], []
    for k in range(max_iter):
        grad = AtA @ x - Aty
        x_new = soft_threshold(x - step * grad, step * lam)
        # 收敛判据
        dx = np.linalg.norm(x_new - x)
        x = x_new
        res = A @ x - y
        obj = 0.5 * np.sum(res ** 2) + lam * np.sum(np.abs(x))
        obj_hist.append(obj)
        sp_hist.append(int(np.sum(np.abs(x) > 1e-10)))
        if dx < tol * (np.linalg.norm(x) + 1e-12):
            break
    return x, dict(obj=obj_hist, sparsity=sp_hist,
                   residual=np.linalg.norm(A @ x - y))


def fista_l1(A, y, lam, x0=None, max_iter=3000, tol=1e-9, restart=True):
    """FISTA 求解 LASSO (Beck-Teboulle 2009).

    加速序列:
        t_0 = 1
        y^k = x^k + ((t_{k-1} - 1) / t_k) (x^k - x^{k-1})
        x^{k+1} = S_{λ/L}(y^k - (1/L) ∇f(y^k))
        t_{k+1} = (1 + √(1 + 4 t_k²)) / 2

    重启策略 (restart): 当 ⟨x^{k+1} - x^k, x^k - x^{k-1}⟩ > 0 时重置 t.
    """
    m, n = A.shape
    if x0 is None:
        x0 = np.zeros(n)
    x = x0.copy()
    x_prev = x0.copy()
    L = lipschitz_constant(A)
    step = 1.0 / L

    AtA = A.T @ A
    Aty = A.T @ y
    t = 1.0
    obj_hist, sp_hist = [], []
    for k in range(max_iter):
        # 外推点
        y_k = x + ((t - 1.0) / max(t, 1.0)) * (x - x_prev)
        grad = AtA @ y_k - Aty
        x_new = soft_threshold(y_k - step * grad, step * lam)
        # 重启判据
        if restart and np.dot(x_new - x, x - x_prev) > 0:
            t = 1.0
        else:
            t = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        dx = np.linalg.norm(x_new - x)
        x_prev = x
        x = x_new
        res = A @ x - y
        obj = 0.5 * np.sum(res ** 2) + lam * np.sum(np.abs(x))
        obj_hist.append(obj)
        sp_hist.append(int(np.sum(np.abs(x) > 1e-10)))
        if dx < tol * (np.linalg.norm(x) + 1e-12):
            break
    return x, dict(obj=obj_hist, sparsity=sp_hist,
                   residual=np.linalg.norm(A @ x - y))


# ----------------------------------------------------------------------
# Heavy-ball 近端 (源自 1115 惯性振子)
# ----------------------------------------------------------------------
def heavy_ball_proximal(A, y, lam, mass=0.1, damping=0.9,
                        max_iter=3000, tol=1e-9):
    """Heavy-ball 惯性近端方法.

    连续动力学:
        m ẍ + γ ẋ + A^T(Ax-y) + λ ∂||x||_1 ∋ 0
    离散化 (显式 Euler):
        v^k = x^k - α (A^T(Ax^k-y))
        x^{k+1} = S_{αλ}(v^k + β(x^k - x^{k-1}))
    参数约束:
        α < 2/L,   β < 1 - √(αL)   (保证收敛)
    """
    m, n = A.shape
    x = np.zeros(n)
    x_prev = np.zeros(n)
    L = lipschitz_constant(A)
    alpha = 0.9 * 2.0 / L  # 安全步长
    beta = min(damping * (1 - np.sqrt(alpha * L)), 0.99)
    beta = max(beta, 0.0)

    AtA = A.T @ A
    Aty = A.T @ y
    obj_hist, sp_hist = [], []
    for k in range(max_iter):
        grad = AtA @ x - Aty
        v = x - alpha * grad
        x_new = soft_threshold(v + beta * (x - x_prev), alpha * lam)
        dx = np.linalg.norm(x_new - x)
        x_prev = x
        x = x_new
        res = A @ x - y
        obj = 0.5 * np.sum(res ** 2) + lam * np.sum(np.abs(x))
        obj_hist.append(obj)
        sp_hist.append(int(np.sum(np.abs(x) > 1e-10)))
        if dx < tol * (np.linalg.norm(x) + 1e-12):
            break
    return x, dict(obj=obj_hist, sparsity=sp_hist,
                   residual=np.linalg.norm(A @ x - y),
                   alpha=alpha, beta=beta)


# ----------------------------------------------------------------------
# 隐式中点稳定化 (源自 1374_unstable_ode)
# ----------------------------------------------------------------------
def implicit_midpoint_proximal(A, y, lam, h, max_iter=2000, newton_iter=5):
    """不稳定 ODE 稳定化: 隐式中点 + 近端分裂.

    考虑梯度流 ẋ = -∇f(x) - λ sign(x).
    当 L 极大时显式格式发散, 采用隐式中点:
        x^{k+1} = x^k + h · g(0.5(x^k + x^{k+1}))
    内层 Newton 求解, 外层施加软阈值.
    """
    m, n = A.shape
    x = np.zeros(n)
    AtA = A.T @ A
    Aty = A.T @ y
    L = lipschitz_constant(A)
    if h > 1.5 / L:
        h = 1.0 / L  # 自适应缩小步长

    obj_hist, sp_hist = [], []
    for k in range(max_iter):
        # Newton 求解中间步 z = 0.5(x + x_new)
        z = x.copy()
        for _ in range(newton_iter):
            grad_z = AtA @ z - Aty
            # Newton 步 (近似 Hessian: A^T A)
            J = np.eye(n) + 0.5 * h * AtA
            rhs = z - x + h * grad_z
            dz = np.linalg.solve(J, -rhs)
            z = z + dz
            if np.linalg.norm(dz) < 1e-10:
                break
        x_new = soft_threshold(2 * z - x, h * lam)
        dx = np.linalg.norm(x_new - x)
        x = x_new
        res = A @ x - y
        obj = 0.5 * np.sum(res ** 2) + lam * np.sum(np.abs(x))
        obj_hist.append(obj)
        sp_hist.append(int(np.sum(np.abs(x) > 1e-10)))
        if dx < 1e-9 * (np.linalg.norm(x) + 1e-12):
            break
    return x, dict(obj=obj_hist, sparsity=sp_hist,
                   residual=np.linalg.norm(A @ x - y), h=h)


# ----------------------------------------------------------------------
# 对偶诊断
# ----------------------------------------------------------------------
def dual_certificate(A, y, x, lam):
    """KKT 对偶证书.

    最优性条件:
        A^T(Ax-y) + λ z = 0,   z ∈ ∂||x||_1
    其中 z_i = sign(x_i) 若 x_i ≠ 0;  |z_i| ≤ 1 若 x_i = 0.
    返回: 最大违反量 max_{i: x_i=0} |z_i| - 1.
    """
    res = A @ x - y
    z = (A.T @ res - 0.0) / lam  # z = -(A^T(Ax-y))/λ
    support = np.abs(x) > 1e-10
    # 在支撑集上 z_i 应等于 sign(x_i)
    support_violation = np.max(np.abs(z[support] - np.sign(x[support]))) \
        if np.any(support) else 0.0
    # 在零元素上 |z_i| ≤ 1
    zero_violation = np.max(np.abs(z[~support]) - 1.0) \
        if np.any(~support) else 0.0
    zero_violation = max(zero_violation, 0.0)
    return float(max(support_violation, zero_violation))
