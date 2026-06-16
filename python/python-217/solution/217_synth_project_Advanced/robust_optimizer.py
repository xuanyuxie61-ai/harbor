"""
robust_optimizer.py
-------------------
鲁棒优化主求解器 —— 综合所有模块的核心优化循环
核心思想：实现多种鲁棒优化算法，求解带不确定约束的 SMB 过程优化。

算法：
    1. 最坏情况鲁棒优化 (WCRO)
        min_x sup_{w in W} F(x, w)
    2. 分布鲁棒优化 (DRO)
        min_x sup_{P in P} E_P[ F(x, w) ]
    3. 机会约束鲁棒优化 (CCRO)
        min_x F(x)
        s.t. P( g_i(x, w) <= 0 ) >= 1 - alpha_i
    4. 均值-方差鲁棒优化 (MVRO)
        min_x E[ F(x, w) ] + beta * Var[ F(x, w) ]

求解器：
    - 次梯度法 (最坏情况)
    - 投影梯度法 (带机会约束)
    - LMC 采样 (分布鲁棒)
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple, Optional, Dict, List
try:
    from uncertainty_set import EllipsoidalUncertaintySet
    from chance_constraints import ChanceConstraint, JointChanceConstraint
    from adjoint_sensitivity import gradient_central, gradient_richardson
    from circulant_kkt import circulant_solve
except ImportError:
    from .uncertainty_set import EllipsoidalUncertaintySet
    from .chance_constraints import ChanceConstraint, JointChanceConstraint
    from .adjoint_sensitivity import gradient_central, gradient_richardson
    from .circulant_kkt import circulant_solve


# =============================================================================
# 鲁棒优化问题定义
# =============================================================================
class RobustOptimizationProblem:
    """
    鲁棒优化问题：
        min_{x in X}  F_robust(x)
        s.t.  g_j(x) <= 0,  j = 1..m
    """

    def __init__(
        self,
        dim: int,
        objective_fn: Callable[[np.ndarray], float],
        worst_case_fn: Callable[[np.ndarray], float],
        constraint_fns: Optional[List[Callable[[np.ndarray], float]]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.dim = dim
        self.objective_fn = objective_fn
        self.worst_case_fn = worst_case_fn
        self.constraint_fns = constraint_fns or []
        if bounds is None:
            self.x_low = -10.0 * np.ones(dim)
            self.x_high = 10.0 * np.ones(dim)
        else:
            self.x_low = np.asarray(bounds[0], dtype=float).ravel()
            self.x_high = np.asarray(bounds[1], dtype=float).ravel()

    def is_feasible(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """检查可行性."""
        x = np.asarray(x, dtype=float).ravel()
        if np.any(x < self.x_low - tol) or np.any(x > self.x_high + tol):
            return False
        for g in self.constraint_fns:
            if g(x) > tol:
                return False
        return True

    def project(self, x: np.ndarray) -> np.ndarray:
        """投影到可行域 (盒约束)."""
        return np.clip(x, self.x_low, self.x_high)


# =============================================================================
# 最坏情况鲁棒优化 (次梯度法)
# =============================================================================
def worst_case_robust_optimize(
    problem: RobustOptimizationProblem,
    uncertainty_set: EllipsoidalUncertaintySet,
    x0: np.ndarray,
    lr: float = 0.01,
    n_iter: int = 500,
    n_samples: int = 20,
    momentum: float = 0.9,
) -> Dict[str, any]:
    """
    最坏情况鲁棒优化 (次梯度法)：
        x_{k+1} = Proj_X( x_k - lr_k * g_k )
    其中 g_k = nabla_x F(x_k, w_k^*),  w_k^* = argmax_{w in W} F(x_k, w).

    返回 dict: {x_opt, f_opt, history, converged}.
    """
    x = problem.project(x0.copy())
    v = np.zeros(problem.dim)  # 动量
    history = []

    for k in range(n_iter):
        # 采样不确定性
        w_samples = uncertainty_set.sample(n_samples)
        # 寻找最坏情况
        f_vals = []
        for w in w_samples:
            f_w = problem.worst_case_fn(np.concatenate([x, w]))
            f_vals.append(f_w)
        f_vals = np.array(f_vals)
        idx_wc = np.argmax(f_vals)
        w_wc = w_samples[idx_wc]

        # 计算梯度
        x_full = np.concatenate([x, w_wc])
        def f_local(z):
            return problem.worst_case_fn(np.concatenate([z, w_wc]))
        g = gradient_central(f_local, x, h=1e-4)

        # 动量更新
        v = momentum * v + (1.0 - momentum) * g
        # 自适应步长
        lr_k = lr / (1.0 + 0.01 * k)
        x = x - lr_k * v
        x = problem.project(x)

        f_val = problem.objective_fn(x)
        history.append({"iter": k, "f": f_val, "x_norm": float(np.linalg.norm(x))})

    return {
        "x_opt": x,
        "f_opt": problem.objective_fn(x),
        "history": history,
        "converged": len(history) > 10 and abs(history[-1]["f"] - history[-10]["f"]) < 1e-6,
    }


# =============================================================================
# 机会约束鲁棒优化 (投影梯度法)
# =============================================================================
def chance_constrained_robust_optimize(
    problem: RobustOptimizationProblem,
    chance_constraints: JointChanceConstraint,
    x0: np.ndarray,
    lr: float = 0.01,
    n_iter: int = 500,
    penalty_weight: float = 100.0,
) -> Dict[str, any]:
    """
    机会约束鲁棒优化 (精确罚函数法)：
        min_x F(x) + mu * max(0, max_j g_j(x))^2
    其中 g_j 为机会约束的确定性等价。
    """
    x = problem.project(x0.copy())
    history = []

    for k in range(n_iter):
        # 目标梯度
        def f_obj(z):
            return problem.objective_fn(z)
        g_obj = gradient_central(f_obj, x, h=1e-4)

        # 机会约束违反
        violations = chance_constraints.evaluate(x)
        max_viol = np.max(violations)
        # 惩罚梯度
        if max_viol > 0.0:
            # 最紧约束的梯度
            idx_tight = np.argmax(violations)
            cc = chance_constraints.constraints[idx_tight]
            g_pen = cc.gradient(x)
            g_total = g_obj + 2.0 * penalty_weight * max_viol * g_pen
        else:
            g_total = g_obj

        # 梯度步
        lr_k = lr / (1.0 + 0.01 * k)
        x = x - lr_k * g_total
        x = problem.project(x)

        f_val = problem.objective_fn(x)
        history.append({
            "iter": k,
            "f": f_val,
            "max_viol": float(max_viol),
            "feasible": max_viol <= 1e-6,
        })

    return {
        "x_opt": x,
        "f_opt": problem.objective_fn(x),
        "history": history,
        "feasible": history[-1]["feasible"],
    }


# =============================================================================
# 均值-方差鲁棒优化
# =============================================================================
def mean_variance_robust_optimize(
    problem: RobustOptimizationProblem,
    uncertainty_set: EllipsoidalUncertaintySet,
    x0: np.ndarray,
    risk_weight: float = 0.5,
    lr: float = 0.01,
    n_iter: int = 500,
    n_samples: int = 50,
) -> Dict[str, any]:
    """
    均值-方差鲁棒优化：
        min_x (1 - beta) E_w[F(x,w)] + beta sup_w F(x,w)
    采用样本平均近似 + 投影梯度。
    """
    x = problem.project(x0.copy())
    history = []

    for k in range(n_iter):
        # 采样
        w_samples = uncertainty_set.sample(n_samples)
        # 计算均值和最坏情况
        f_vals = np.array([
            problem.worst_case_fn(np.concatenate([x, w]))
            for w in w_samples
        ])
        f_mean = np.mean(f_vals)
        f_wc = np.max(f_vals)
        f_mv = (1.0 - risk_weight) * f_mean + risk_weight * f_wc

        # 梯度
        def f_mv_fn(z):
            vals = np.array([
                problem.worst_case_fn(np.concatenate([z, w]))
                for w in w_samples
            ])
            return (1.0 - risk_weight) * np.mean(vals) + risk_weight * np.max(vals)

        g = gradient_central(f_mv_fn, x, h=1e-4)

        # 更新
        lr_k = lr / (1.0 + 0.01 * k)
        x = x - lr_k * g
        x = problem.project(x)

        history.append({"iter": k, "f": f_mv, "mean": float(f_mean), "wc": float(f_wc)})

    return {
        "x_opt": x,
        "f_opt": history[-1]["f"],
        "history": history,
    }


# =============================================================================
# KKT 系统求解 (用于 PDE 约束)
# =============================================================================
def solve_kkt_system(
    grad_obj: np.ndarray,
    constraint_jacobian: np.ndarray,
    constraint_vals: np.ndarray,
    mu: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解 KKT 系统：
        [ I   J^T ] [ dx ]   [ -grad_f ]
        [ J  -mu I] [ dy ] = [ -c       ]
    其中 J 为约束 Jacobian，c 为约束值。
    """
    n = grad_obj.size
    m = constraint_vals.size
    KKT = np.block([
        [np.eye(n), constraint_jacobian.T],
        [constraint_jacobian, -mu * np.eye(m)],
    ])
    rhs = np.concatenate([-grad_obj, -constraint_vals])
    try:
        sol = np.linalg.solve(KKT + 1e-10 * np.eye(n + m), rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
    return sol[:n], sol[n:]


# ----------------------------------------------------------------------
# 自检 (不在此模块运行，由 main.py 统一调用)
# ----------------------------------------------------------------------
