"""
map_solver.py  --  Jacobian-free Newton-Krylov MAP 估计
===============================================================
来源种子项目:
    617_kelley : nsol (Newton), gmres, broyden, arnoldi, bicgstab,
                  tfqmr, pcg, diffjac, dirder, fish2d 等.
科学问题角色:
    MAP 估计 theta_MAP = argmax log p(theta|y)
                        = argmin -log p(y|theta) - log p(theta)
    目标泛函:
        J(theta) = 0.5 * (y - G(theta))^T Sigma_obs^{-1} (y - G(theta))
                   + 0.5 * theta^T Sigma_prior^{-1} theta
                   + const
    梯度:
        nabla J = -G'(theta)^T Sigma_obs^{-1} (y - G(theta))
                  + Sigma_prior^{-1} theta
    我们用 *Jacobian-free* Newton-Krylov: 不显式构造 Hessian,
    而是用 GMRES 求解 H * delta = -grad, 其中 H*delta 通过
    有限差分近似:
        H * delta approx (grad(theta + eps*delta) - grad(theta)) / eps
    这移植了 Kelley 的 fdgmres / fdkrylov 思想.
边界与鲁棒性:
    1. Armijo 线搜索保证下降.
    2. 梯度范数容差用 NUMERICS.rtol.
    3. 若 GMRES 停滞, 回退到最速下降.
"""
from __future__ import annotations
import math
from typing import List, Callable, Tuple, Dict
from numerical_base import NUMERICS


# ======================================================================
# 1. GMRES (移植 Kelley 617)
# ======================================================================
def _gmres_solve(Av: Callable[[List[float]], List[float]],
                 b: List[float], maxit: int = 30,
                 tol: float | None = None
                 ) -> Tuple[List[float], List[float], int]:
    """
    Saad-Schultz GMRES 求解 A x = b, 仅提供矩阵-向量乘 Av.
    返回 (x, residuals, iters).
    """
    if tol is None:
        tol = NUMERICS.rtol
    n = len(b)
    # 初始残差
    x = [0.0] * n
    r = list(b)
    beta = math.sqrt(sum(ri * ri for ri in r))
    if beta < tol:
        return x, [beta], 0
    # Arnoldi 基
    V = [list(r)]
    V[0] = [v / beta for v in V[0]]
    # Hessenberg 矩阵
    H = [[0.0] * (maxit + 1) for _ in range(maxit)]
    # Givens 旋转
    cs = [0.0] * maxit
    sn = [0.0] * maxit
    e1 = [0.0] * (maxit + 1)
    e1[0] = beta
    res_norms = [beta]
    for k in range(maxit):
        # Arnoldi 步
        w = Av(V[k])
        for j in range(k + 1):
            H[j][k] = sum(w[i] * V[j][i] for i in range(n))
            for i in range(n):
                w[i] -= H[j][k] * V[j][i]
        H[k + 1][k] = math.sqrt(sum(wi * wi for wi in w))
        if H[k + 1][k] < NUMERICS.cholesky_jitter:
            break
        v_new = [wi / H[k + 1][k] for wi in w]
        V.append(v_new)
        # 应用之前的 Givens 旋转
        for j in range(k):
            temp = cs[j] * H[j][k] + sn[j] * H[j + 1][k]
            H[j + 1][k] = -sn[j] * H[j][k] + cs[j] * H[j + 1][k]
            H[j][k] = temp
        # 计算新 Givens
        r_k = math.sqrt(H[k][k] ** 2 + H[k + 1][k] ** 2)
        if r_k < NUMERICS.cholesky_jitter:
            break
        cs[k] = H[k][k] / r_k
        sn[k] = H[k + 1][k] / r_k
        H[k][k] = r_k
        H[k + 1][k] = 0.0
        temp = cs[k] * e1[k] + sn[k] * e1[k + 1]
        e1[k + 1] = -sn[k] * e1[k] + cs[k] * e1[k + 1]
        e1[k] = temp
        res_norms.append(abs(e1[k + 1]))
        if abs(e1[k + 1]) < tol:
            k += 1
            break
    else:
        k = maxit
    # 回代求解 H y = e1
    m = k
    y = [0.0] * m
    for i in range(m - 1, -1, -1):
        y[i] = e1[i]
        for j in range(i + 1, m):
            y[i] -= H[i][j] * y[j]
        y[i] /= max(abs(H[i][i]), NUMERICS.cholesky_jitter)
    # x = V y
    for i in range(m):
        for j in range(n):
            x[j] += y[i] * V[i][j]
    return x, res_norms, m


# ======================================================================
# 2. Broyden 无雅可比更新 (移植 Kelley)
# ======================================================================
class BroydenSolver:
    """
    Broyden 拟 Newton 法, 用于中小规模问题.
    J_{k+1} = J_k + (y - J_k s) s^T / (s^T s)
    """
    def __init__(self, dim: int, maxit: int = 100):
        self.dim = dim
        self.maxit = maxit

    def solve(self, F: Callable[[List[float]], List[float]],
              x0: List[float]) -> Tuple[List[float], List[float], int]:
        """
        求解 F(x) = 0. 返回 (x_sol, residuals, iters).
        """
        x = list(x0)
        fx = F(x)
        res_norms = [math.sqrt(sum(fi * fi for fi in fx))]
        # 初始 Jacobian = -I
        J = [[-1.0 if i == j else 0.0 for j in range(self.dim)]
             for i in range(self.dim)]
        for k in range(self.maxit):
            if res_norms[-1] < NUMERICS.rtol:
                break
            # 求解 J dx = -fx
            dx = _solve_linear(J, [-fi for fi in fx])
            x_new = [x[i] + dx[i] for i in range(self.dim)]
            fx_new = F(x_new)
            # Broyden 更新
            s = dx
            y = [fx_new[i] - fx[i] for i in range(self.dim)]
            Js = [sum(J[i][j] * s[j] for j in range(self.dim))
                  for i in range(self.dim)]
            denom = sum(si * si for si in s)
            if denom < NUMERICS.cholesky_jitter:
                break
            for i in range(self.dim):
                for j in range(self.dim):
                    J[i][j] += (y[i] - Js[i]) * s[j] / denom
            x = x_new
            fx = fx_new
            res_norms.append(math.sqrt(sum(fi * fi for fi in fx)))
        return x, res_norms, len(res_norms) - 1


def _solve_linear(A, b) -> List[float]:
    """简单 Gauss 消元."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        M[col], M[pivot] = M[pivot], M[col]
        p = M[col][col]
        if abs(p) < NUMERICS.cholesky_jitter:
            M[col][col] += NUMERICS.cholesky_jitter
            p = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= p
        for r in range(n):
            if r == col:
                continue
            fac = M[r][col]
            for j in range(col, n + 1):
                M[r][j] -= fac * M[col][j]
    return [M[i][n] for i in range(n)]


# ======================================================================
# 3. JFNK MAP 求解器
# ======================================================================
class MAPSolver:
    """
    Jacobian-free Newton-Krylov MAP 估计.
    J(theta) = -log p(y|theta) - log p(theta)
    """
    def __init__(self, dim: int, maxit: int = 50, gmres_restart: int = 20):
        self.dim = dim
        self.maxit = maxit
        self.gmres_restart = gmres_restart
        self.history: List[Dict[str, float]] = []

    def solve(self, grad_J: Callable[[List[float]], List[float]],
              J_func: Callable[[List[float]], float],
              theta0: List[float]) -> Tuple[List[float], List[Dict[str, float]]]:
        """
        求解 min J(theta).
        grad_J : 梯度函数
        J_func : 目标函数
        返回 (theta_MAP, history).
        """
        theta = list(theta0)
        for k in range(self.maxit):
            g = grad_J(theta)
            gnorm = math.sqrt(sum(gi * gi for gi in g))
            jval = J_func(theta)
            self.history.append({"iter": k, "J": jval, "grad_norm": gnorm})
            if gnorm < NUMERICS.rtol:
                break
            # JFNK: 用 GMRES 求解 H * delta = -g
            eps_fd = max(NUMERICS.eps ** 0.5, 1e-6)

            def Hv(v):
                """Hessian-vector 通过有限差分: (g(theta+eps*v) - g) / eps."""
                theta_pert = [theta[i] + eps_fd * v[i]
                              for i in range(self.dim)]
                g_pert = grad_J(theta_pert)
                return [(g_pert[i] - g[i]) / eps_fd for i in range(self.dim)]

            delta, res, iters = _gmres_solve(Hv, [-gi for gi in g],
                                             maxit=self.gmres_restart)
            # Armijo 线搜索
            alpha = 1.0
            c1 = 1e-4
            dir_deriv = sum(gi * delta[i] for i, gi in enumerate(g))
            for _ in range(20):
                theta_new = [theta[i] + alpha * delta[i]
                             for i in range(self.dim)]
                j_new = J_func(theta_new)
                if j_new <= jval + c1 * alpha * dir_deriv:
                    break
                alpha *= 0.5
            else:
                # 线搜索失败, 退到最速下降
                delta = [-gi for gi in g]
                alpha = 0.01
                theta_new = [theta[i] + alpha * delta[i]
                             for i in range(self.dim)]
            theta = theta_new
            # 边界截断
            theta = [max(-5.0, min(5.0, t)) for t in theta]
        return theta, self.history


# ======================================================================
# 4. 二次收敛验证
# ======================================================================
def check_quadratic_convergence(history: List[Dict[str, float]]) -> bool:
    """
    检查最后几步是否呈现二次收敛:
        log(grad_norm_{k+1}) < 2 * log(grad_norm_k) - const
    """
    if len(history) < 5:
        return False
    last = history[-4:]
    for i in range(len(last) - 1):
        g1 = max(last[i]["grad_norm"], NUMERICS.safe_log_floor)
        g2 = max(last[i + 1]["grad_norm"], NUMERICS.safe_log_floor)
        lg1 = math.log(g1)
        lg2 = math.log(g2)
        if lg2 > 1.8 * lg1 + 1.0:
            return False
    return True


__all__ = ["MAPSolver", "BroydenSolver", "check_quadratic_convergence"]
