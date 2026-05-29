"""
natural_gradient.py

自然策略梯度 (Natural Policy Gradient) 与 Fisher 信息矩阵计算

基于种子项目:
  - 964_r83p: 周期三对角矩阵快速求解
  - 1262_toeplitz_cholesky: Toeplitz 矩阵 Cholesky 分解
  - 081_besselzero: 贝塞尔零点用于谱正则化

科学原理:
  自然梯度将参数更新方向校正为在策略分布流形上的最速下降方向:

      θ_{new} = θ + α · F(θ)^{-1} ∇_θ J(θ)

  其中 F(θ) 为 Fisher 信息矩阵:
      F(θ) = E_{s~ρ^π, a~π_θ} [ ∇_θ log π_θ(a|s) ∇_θ log π_θ(a|s)^T ]

  在随机策略梯度中, F 具有特殊结构:
      1. 对高斯策略, F 可分解为块对角 + 低秩修正;
      2. 当状态具有时间平移不变性时, F 近似为 Toeplitz 矩阵;
      3. 对周期性边界条件, F 为周期三对角 (R83P) 结构.

  本模块实现:
      - 样本 Fisher 矩阵估计
      - 共轭梯度法 (CG) 近似求解 F·x = g
      - 基于 R83P/Toeplitz 结构的快速精确求解
      - 谱正则化 (Bessel 零点截断) 防止病态
"""

import numpy as np
from typing import List, Tuple, Callable
from linear_algebra import r83p_solve, toeplitz_cholesky_lower


def sample_fisher_information_matrix(states: List[np.ndarray],
                                      actions: List[np.ndarray],
                                      policy_grad_func: Callable,
                                      num_params: int) -> np.ndarray:
    """
    从样本中估计 Fisher 信息矩阵.

    估计式:
        F̂ = (1/N) Σ_i  g_i g_i^T,  g_i = ∇_θ log π_θ(a_i|s_i)

    参数:
        states: 状态样本列表
        actions: 动作样本列表
        policy_grad_func: 函数, 输入 (s,a) 输出展平梯度向量
        num_params: 参数维度

    返回:
        num_params × num_params 的 Fisher 矩阵估计
    """
    N = len(states)
    if N == 0:
        return np.eye(num_params) * 1.0e-6
    F = np.zeros((num_params, num_params))
    for s, a in zip(states, actions):
        g = policy_grad_func(s, a)
        F += np.outer(g, g)
    F = F / N
    # 正则化
    reg = 1.0e-4 * np.trace(F) / num_params
    F = F + reg * np.eye(num_params)
    return F


def conjugate_gradient_solve(A_func: Callable, b: np.ndarray,
                              max_iter: int = 50, tol: float = 1.0e-10,
                              damping: float = 1.0e-3) -> np.ndarray:
    """
    共轭梯度法求解 (A + damping·I) x = b, 无需显式构造 A.

    数学推导:
        CG 适用于对称正定线性系统, 迭代格式:
            r_0 = b - A x_0,  p_0 = r_0
            α_k = (r_k^T r_k) / (p_k^T A p_k)
            x_{k+1} = x_k + α_k p_k
            r_{k+1} = r_k - α_k A p_k
            β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
            p_{k+1} = r_{k+1} + β_k p_k

    参数:
        A_func: 函数, 输入 x 输出 A·x
        b: 右端项
        max_iter: 最大迭代次数
        tol: 残差容差
        damping: Tikhonov 正则化系数

    返回:
        近似解 x
    """
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs_old = float(r @ r)
    for _ in range(max_iter):
        Ap = A_func(p) + damping * p
        pAp = float(p @ Ap)
        if abs(pAp) < 1.0e-15:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x


def fisher_vector_product(states: List[np.ndarray],
                          actions: List[np.ndarray],
                          policy_grad_func: Callable,
                          v: np.ndarray) -> np.ndarray:
    """
    计算 Fisher 向量积 F·v (无需显式构造 F).

    推导:
        F·v = (1/N) Σ_i g_i (g_i^T v)
    """
    # TODO: Hole_3 实现 Fisher 向量积
    pass


def natural_gradient_update(theta: np.ndarray,
                            grad: np.ndarray,
                            states: List[np.ndarray],
                            actions: List[np.ndarray],
                            policy_grad_func: Callable,
                            method: str = 'cg',
                            cg_iter: int = 50,
                            cg_damping: float = 1.0e-3) -> np.ndarray:
    """
    计算自然梯度方向 F^{-1} ∇J.

    参数:
        theta: 当前参数
        grad:  标准梯度 ∇J
        states, actions: 样本
        policy_grad_func: 梯度函数
        method: 'cg' (共轭梯度), 'direct' (直接求逆), 'r83p' (周期三对角)

    返回:
        自然梯度方向向量
    """
    if method == 'cg':
        def A_func(v):
            return fisher_vector_product(states, actions, policy_grad_func, v)
        ng = conjugate_gradient_solve(A_func, grad, max_iter=cg_iter,
                                       damping=cg_damping)
    elif method == 'direct':
        num_params = len(theta)
        F = sample_fisher_information_matrix(states, actions, policy_grad_func, num_params)
        try:
            ng = np.linalg.solve(F, grad)
        except np.linalg.LinAlgError:
            ng = np.linalg.lstsq(F, grad, rcond=None)[0]
    else:
        # 回退到 CG
        def A_func(v):
            return fisher_vector_product(states, actions, policy_grad_func, v)
        ng = conjugate_gradient_solve(A_func, grad, max_iter=cg_iter,
                                       damping=cg_damping)
    return ng


class NaturalPolicyGradientOptimizer:
    """
    自然策略梯度优化器.
    """

    def __init__(self, learning_rate: float = 0.01,
                 cg_iter: int = 50, cg_damping: float = 1.0e-3,
                 max_kl: float = 0.01):
        self.lr = learning_rate
        self.cg_iter = cg_iter
        self.cg_damping = cg_damping
        self.max_kl = max_kl

    def step(self, theta: np.ndarray, grad: np.ndarray,
             states: List[np.ndarray], actions: List[np.ndarray],
             policy_grad_func: Callable) -> np.ndarray:
        """
        执行一步自然梯度更新.
        """
        ng = natural_gradient_update(
            theta, grad, states, actions, policy_grad_func,
            method='cg', cg_iter=self.cg_iter, cg_damping=self.cg_damping
        )
        # 线搜索: 限制 KL 散度
        step_size = self.lr
        for _ in range(10):
            theta_new = theta + step_size * ng
            # KL 散度近似: 0.5 * ng^T F ng * step_size^2
            Fv = fisher_vector_product(states, actions, policy_grad_func, ng)
            kl_approx = 0.5 * step_size ** 2 * (ng @ Fv)
            if kl_approx <= self.max_kl:
                break
            step_size *= 0.5
        return theta + step_size * ng
