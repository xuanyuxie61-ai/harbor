"""
constrained_calibration.py  --  带安全屏障的约束贝叶斯校准
===============================================================
来源种子项目:
    1260_Masonniu_Energy-Sufficient-Safe-Centralized-Control :
        MOSEK Fusion 优化 + 控制障碍函数 (CBF) 保证安全不变性.
科学问题角色:
    某些参数受物理约束 (如扩散系数必须正, 反应速率有上限).
    我们在后验中引入 *对数屏障*:
        log p_constrained(theta|y) = log p(theta|y)
                                      + sum_i mu_i * log(h_i(theta))
    其中 h_i(theta) >= 0 定义可行域.
    这等价于 CBF 思想: 保证后验样本永不越界.
    对约束优化 (MAP), 用 MOSEK 风格的 *二阶锥松弛*:
        min J(theta) + sum mu_i * (-log h_i(theta))
核心公式:
    屏障参数更新: mu_{k+1} = mu_k / barrier_decay
    内层: Newton 求解 unconstrained J + barrier.
    收敛条件: ||grad|| < tol AND h_i(theta) > 0 for all i.
"""
from __future__ import annotations
import math
from typing import List, Callable, Tuple, Dict
from numerical_base import NUMERICS


# ======================================================================
# 1. 约束定义 (CBF 风格)
# ======================================================================
class SafetyConstraint:
    """
    h(theta) >= 0 形式的安全约束.
    例: theta[0] >= -5  ->  h(theta) = theta[0] + 5
    """
    def __init__(self, name: str,
                 h_func: Callable[[List[float]], float],
                 grad_h: Callable[[List[float]], List[float]]):
        self.name = name
        self.h_func = h_func
        self.grad_h = grad_h

    def evaluate(self, theta: List[float]) -> float:
        return self.h_func(theta)

    def gradient(self, theta: List[float]) -> List[float]:
        return self.grad_h(theta)

    def is_feasible(self, theta: List[float], tol: float = 0.0) -> bool:
        return self.evaluate(theta) >= tol


def build_default_constraints(dim: int) -> List[SafetyConstraint]:
    """
    默认约束集: 对每个参数 theta_i in [-5, 5].
    h_{2i}(theta) = theta_i + 5
    h_{2i+1}(theta) = 5 - theta_i
    """
    constraints = []
    for i in range(dim):
        # theta_i >= -5
        constraints.append(SafetyConstraint(
            f"theta_{i}>=-5",
            h_func=lambda t, _i=i: t[_i] + 5.0,
            grad_h=lambda t, _i=i, _d=dim: [
                1.0 if j == _i else 0.0 for j in range(_d)]
        ))
        # theta_i <= 5
        constraints.append(SafetyConstraint(
            f"theta_{i}<=5",
            h_func=lambda t, _i=i: 5.0 - t[_i],
            grad_h=lambda t, _i=i, _d=dim: [
                -1.0 if j == _i else 0.0 for j in range(_d)]
        ))
    return constraints


# ======================================================================
# 2. 屏障函数
# ======================================================================
class LogBarrier:
    """
    对数屏障: B(theta) = sum_i -log(h_i(theta))
    梯度: grad B = sum_i -grad h_i / h_i
    """
    def __init__(self, constraints: List[SafetyConstraint],
                 mu: float = 1.0):
        self.constraints = constraints
        self.mu = mu

    def evaluate(self, theta: List[float]) -> float:
        s = 0.0
        for c in self.constraints:
            h = c.evaluate(theta)
            if h <= NUMERICS.safe_log_floor:
                return 1e10  # 不可行惩罚
            s += -math.log(h)
        return self.mu * s

    def gradient(self, theta: List[float]) -> List[float]:
        dim = len(theta)
        grad = [0.0] * dim
        for c in self.constraints:
            h = c.evaluate(theta)
            if h <= NUMERICS.safe_log_floor:
                # 返回大梯度推向可行域
                for i in range(dim):
                    grad[i] += 1e6
                continue
            gh = c.gradient(theta)
            for i in range(dim):
                grad[i] -= self.mu * gh[i] / h
        return grad

    def update_mu(self, decay: float = 0.5) -> None:
        """减小屏障参数, 逼近原始约束."""
        self.mu = max(NUMERICS.cholesky_jitter, self.mu * decay)


# ======================================================================
# 3. 约束 MAP 求解 (内点法)
# ======================================================================
class ConstrainedMAPSolver:
    """
    内点法求解约束 MAP:
        min J(theta) + B(theta)
        s.t. h_i(theta) >= 0
    外层: mu 递减; 内层: 无约束 Newton.
    """
    def __init__(self, dim: int, max_outer: int = 10,
                 max_inner: int = 50, barrier_decay: float = 0.5,
                 mu_init: float = 1.0):
        self.dim = dim
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.barrier_decay = barrier_decay
        self.mu_init = mu_init
        self.constraints = build_default_constraints(dim)
        self.barrier = LogBarrier(self.constraints, mu=mu_init)
        self.history: List[Dict[str, float]] = []

    def solve(self, J_func: Callable[[List[float]], float],
              grad_J: Callable[[List[float]], List[float]],
              theta0: List[float]) -> Tuple[List[float],
                                             List[Dict[str, float]]]:
        """
        求解约束 MAP.
        """
        theta = list(theta0)
        # 投影到可行域
        theta = self._project_feasible(theta)
        for outer in range(self.max_outer):
            # 内层: 无约束 Newton 求解 J + barrier
            def combined_J(t):
                return J_func(t) + self.barrier.evaluate(t)

            def combined_grad(t):
                gj = grad_J(t)
                gb = self.barrier.gradient(t)
                return [gj[i] + gb[i] for i in range(self.dim)]

            theta, inner_hist = _simple_newton(
                combined_J, combined_grad, theta,
                maxit=self.max_inner)
            # 记录
            jval = J_func(theta)
            gnorm = math.sqrt(sum(gi * gi
                                  for gi in grad_J(theta)))
            min_h = min(c.evaluate(theta) for c in self.constraints)
            self.history.append({
                "outer": outer, "J": jval, "grad_norm": gnorm,
                "min_h": min_h, "mu": self.barrier.mu
            })
            # 更新屏障
            self.barrier.update_mu(self.barrier_decay)
            # 收敛判断
            if gnorm < NUMERICS.rtol and min_h > 0:
                break
        return theta, self.history

    def _project_feasible(self, theta: List[float]) -> List[float]:
        """简单截断到 [-5+eps, 5-eps]."""
        eps = 0.1
        return [max(-5.0 + eps, min(5.0 - eps, t)) for t in theta]


def _simple_newton(J_func, grad_J, theta0, maxit=50):
    """简单 Newton 带 Armijo 线搜索."""
    from map_solver import _gmres_solve
    dim = len(theta0)
    theta = list(theta0)
    history = []
    for k in range(maxit):
        g = grad_J(theta)
        gnorm = math.sqrt(sum(gi * gi for gi in g))
        jval = J_func(theta)
        history.append({"iter": k, "J": jval, "grad_norm": gnorm})
        if gnorm < NUMERICS.rtol:
            break
        # 近似 Hessian-vector
        eps_fd = max(NUMERICS.eps ** 0.5, 1e-6)

        def Hv(v):
            theta_pert = [theta[i] + eps_fd * v[i] for i in range(dim)]
            g_pert = grad_J(theta_pert)
            return [(g_pert[i] - g[i]) / eps_fd for i in range(dim)]

        delta, _, _ = _gmres_solve(Hv, [-gi for gi in g], maxit=15)
        # Armijo
        alpha = 1.0
        c1 = 1e-4
        dir_deriv = sum(gi * delta[i] for i, gi in enumerate(g))
        for _ in range(15):
            theta_new = [theta[i] + alpha * delta[i] for i in range(dim)]
            j_new = J_func(theta_new)
            if j_new <= jval + c1 * alpha * dir_deriv:
                break
            alpha *= 0.5
        else:
            alpha = 0.01
            theta_new = [theta[i] + alpha * (-g[i]) for i in range(dim)]
        theta = theta_new
    return theta, history


# ======================================================================
# 4. 可行性检查
# ======================================================================
def check_feasibility(theta: List[float],
                      constraints: List[SafetyConstraint]) -> Dict[str, bool]:
    """返回每个约束的可行性."""
    return {c.name: c.is_feasible(theta) for c in constraints}


__all__ = ["ConstrainedMAPSolver", "SafetyConstraint",
           "LogBarrier", "build_default_constraints",
           "check_feasibility"]
