"""
quasi_newton.py
===============
拟 Newton 法与线搜索 —— 无约束优化的核心引擎

融合种子项目:
  - 210_continuation: Newton 迭代, 增广系统
  - 572_ill_bvp: 病态问题处理, 收敛诊断
  - 1286_Shukti042: 余弦相似度作为目标/度量

核心公式:
  1. Newton 步: x_{k+1} = x_k - H_k^{-1} ∇f(x_k)
  2. BFGS 更新: H_{k+1} = (I - ρs y^T) H_k (I - ρy s^T) + ρ s s^T
     其中 s = x_{k+1} - x_k, y = ∇f_{k+1} - ∇f_k, ρ = 1/(y^T s)
  3. Wolfe 条件:
     充分下降: f(x_k + αp_k) ≤ f(x_k) + c₁α∇f_k^T p_k
     曲率条件: ∇f(x_k + αp_k)^T p_k ≥ c₂∇f_k^T p_k
  4. Armijo 回溯: 找最小 α = β^j 使 f(x + αp) ≤ f(x) + c₁α∇f^T p
  5. L-BFGS 两循环递推算子: O(mn) 存储替代 O(n²)
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict
import math


# ---------------------------------------------------------------------------
# 1. 线搜索 (Wolfe 条件)
# ---------------------------------------------------------------------------

def armijo_backtrack(f: Callable, x: np.ndarray, grad: np.ndarray,
                     direction: np.ndarray, alpha_init: float = 1.0,
                     c1: float = 1e-4, beta: float = 0.5,
                     max_iter: int = 50,
                     f_val: Optional[float] = None) -> Tuple[float, float]:
    """Armijo 回溯线搜索.
    找 α = β^j · α₀ 使:
      f(x + αp) ≤ f(x) + c₁·α·∇f^T·p  (充分下降条件)

    参数:
      c1 ∈ (0,1): 充分下降参数 (通常 1e-4)
      beta ∈ (0,1): 收缩因子 (通常 0.5)

    返回 (alpha, f_new).
    """
    if f_val is None:
        f_val = f(x)
    df_dir = np.dot(grad, direction)

    # 确保下降方向
    if df_dir > 0:
        direction = -grad
        df_dir = np.dot(grad, direction)

    alpha = alpha_init
    for _ in range(max_iter):
        x_new = x + alpha * direction
        f_new = f(x_new)
        # Armijo 条件
        if f_new <= f_val + c1 * alpha * df_dir:
            return alpha, f_new
        alpha *= beta

    return alpha, f_new


def wolfe_line_search(f: Callable, grad_f: Callable,
                      x: np.ndarray, direction: np.ndarray,
                      alpha_init: float = 1.0,
                      c1: float = 1e-4, c2: float = 0.9,
                      max_iter: int = 30) -> Tuple[float, float, np.ndarray]:
    """强 Wolfe 条件线搜索 (Nocedal & Wright, Algorithm 3.5).

    强 Wolfe 条件:
      1. f(x + αp) ≤ f(x) + c₁α∇f^T p  (充分下降)
      2. |∇f(x + αp)^T p| ≤ c₂|∇f^T p|  (曲率条件)

    使用区间缩放 + 三次插值.

    返回 (alpha, f_new, grad_new).
    """
    f0 = f(x)
    g0 = grad_f(x)
    dg0 = np.dot(g0, direction)

    if dg0 > 0:
        direction = -g0
        dg0 = np.dot(g0, direction)

    alpha = alpha_init
    alpha_lo, alpha_hi = 0.0, np.inf
    f_lo = f0
    f_new = f0
    g_new = g0.copy()

    for i in range(max_iter):
        x_new = x + alpha * direction
        f_new = f(x_new)
        g_new = grad_f(x_new)

        # 检查充分下降
        if f_new > f0 + c1 * alpha * dg0 or (i > 0 and f_new >= f_lo):
            alpha_hi = alpha
            alpha = 0.5 * (alpha_lo + alpha_hi)
            continue

        dg_new = np.dot(g_new, direction)

        # 检查曲率条件
        if abs(dg_new) <= c2 * abs(dg0):
            return alpha, f_new, g_new

        if dg_new >= 0:
            alpha_hi = alpha
        else:
            alpha_lo = alpha
            f_lo = f_new

        if alpha_hi < np.inf:
            alpha = 0.5 * (alpha_lo + alpha_hi)
        else:
            alpha *= 2.0

    return alpha, f_new, g_new


# ---------------------------------------------------------------------------
# 2. 纯 Newton 法 (源自 210_continuation/newton)
# ---------------------------------------------------------------------------

def newton_method(f: Callable, grad_f: Callable, hess_f: Callable,
                  x0: np.ndarray, tol: float = 1e-10,
                  max_iter: int = 200,
                  use_line_search: bool = True) -> Dict:
    """Newton 法求解 min f(x).
    源自 210_continuation/newton.

    算法:
      for k = 0, 1, ...
        解 H_k d_k = -g_k  (Newton 步)
        α_k = 线搜索
        x_{k+1} = x_k + α_k d_k

    局部二次收敛: ||x_{k+1} - x*|| ≤ C||x_k - x*||².
    但需 Hessian 正定, 否则方向可能非下降.

    返回 dict: {x, f_val, grad_norm, iterations, converged, history}.
    """
    x = x0.copy().astype(float)
    n = len(x)
    history = []

    for k in range(max_iter):
        fk = f(x)
        gk = grad_f(x)
        gnorm = np.linalg.norm(gk)
        history.append({"iter": k, "f": fk, "gnorm": gnorm})

        if gnorm < tol:
            return {"x": x, "f_val": fk, "grad_norm": gnorm,
                    "iterations": k, "converged": True, "history": history}

        Hk = hess_f(x)

        # Hessian 修正 (确保正定)
        from .linalg_solver import hessian_modification
        Hk_mod = hessian_modification(Hk, delta=1e-8)

        try:
            direction = np.linalg.solve(Hk_mod, -gk)
        except np.linalg.LinAlgError:
            direction = -gk  # 退化为梯度下降

        # 验证下降方向
        if np.dot(gk, direction) > 0:
            direction = -gk

        if use_line_search:
            alpha, fk_new = armijo_backtrack(f, x, gk, direction, f_val=fk)
        else:
            alpha = 1.0
            fk_new = f(x + alpha * direction)

        x = x + alpha * direction

    fk = f(x)
    gk = grad_f(x)
    return {"x": x, "f_val": fk, "grad_norm": np.linalg.norm(gk),
            "iterations": max_iter, "converged": False, "history": history}


# ---------------------------------------------------------------------------
# 3. BFGS 拟 Newton 法
# ---------------------------------------------------------------------------

def bfgs(f: Callable, grad_f: Callable, x0: np.ndarray,
         tol: float = 1e-10, max_iter: int = 500) -> Dict:
    """BFGS 拟 Newton 法 (Broyden-Fletcher-Goldfarb-Shanno).

    核心: 用秩-2 修正逼近逆 Hessian:
      H_{k+1} = (I - ρ_k s_k y_k^T) H_k (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
      其中 s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k, ρ_k = 1/(y_k^T s_k)

    全局收敛性: 对凸函数, BFGS 超线性收敛.
    对非凸函数, 需 Wolfe 线搜索保证 y_k^T s_k > 0 (曲率条件).

    返回 dict: {x, f_val, grad_norm, iterations, converged, history}.
    """
    x = x0.copy().astype(float)
    n = len(x)
    H = np.eye(n)  # 初始逆 Hessian 近似

    gk = grad_f(x)
    fk = f(x)
    history = []

    for k in range(max_iter):
        gnorm = np.linalg.norm(gk)
        history.append({"iter": k, "f": fk, "gnorm": gnorm})

        if gnorm < tol:
            return {"x": x, "f_val": fk, "grad_norm": gnorm,
                    "iterations": k, "converged": True, "history": history}

        direction = -H @ gk

        # 确保下降方向
        if np.dot(gk, direction) > -1e-10 * gnorm:
            direction = -gk
            H = np.eye(n)  # 重置

        alpha, fk_new, gk_new = wolfe_line_search(f, grad_f, x, direction)

        sk = alpha * direction
        yk = gk_new - gk
        skyk = np.dot(sk, yk)

        x = x + sk
        fk = fk_new

        # BFGS 更新 (仅当曲率条件满足)
        if skyk > 1e-10 * np.linalg.norm(sk) * np.linalg.norm(yk):
            rho = 1.0 / skyk
            I = np.eye(n)
            V = I - rho * np.outer(sk, yk)
            H = V @ H @ V.T + rho * np.outer(sk, sk)

        gk = gk_new

    return {"x": x, "f_val": fk, "grad_norm": np.linalg.norm(gk),
            "iterations": max_iter, "converged": False, "history": history}


# ---------------------------------------------------------------------------
# 4. L-BFGS (有限内存 BFGS)
# ---------------------------------------------------------------------------

def lbfgs(f: Callable, grad_f: Callable, x0: np.ndarray,
          m: int = 10, tol: float = 1e-10,
          max_iter: int = 1000) -> Dict:
    """L-BFGS 有限内存拟 Newton 法 (Nocedal, 1980).

    存储最近 m 对 {s_k, y_k}, 通过两循环递归计算方向:
      O(mn) 存储, O(mn) 每步计算, 替代 BFGS 的 O(n²).

    两循环递归 (Nocedal & Wright, Algorithm 7.4):
      q = g_k
      for i = k-1, ..., k-m:
        α_i = ρ_i s_i^T q
        q = q - α_i y_i
      r = H_0^0 q  (初始缩放)
      for i = k-m, ..., k-1:
        β = ρ_i y_i^T r
        r = r + s_i (α_i - β)
      d_k = -r

    特别适合高维优化问题 (n > 1000).
    """
    x = x0.copy().astype(float)
    n = len(x)

    gk = grad_f(x)
    fk = f(x)
    history = []

    s_list = []
    y_list = []
    rho_list = []

    for k in range(max_iter):
        gnorm = np.linalg.norm(gk)
        history.append({"iter": k, "f": fk, "gnorm": gnorm})

        if gnorm < tol:
            return {"x": x, "f_val": fk, "grad_norm": gnorm,
                    "iterations": k, "converged": True, "history": history}

        # 两循环递归计算方向
        direction = _lbfgs_direction(gk, s_list, y_list, rho_list, m)

        # 确保下降方向
        if np.dot(gk, direction) > 0:
            direction = -gk

        alpha, fk_new, gk_new = wolfe_line_search(f, grad_f, x, direction)

        sk = alpha * direction
        yk = gk_new - gk
        skyk = np.dot(sk, yk)

        # 更新历史
        if len(s_list) >= m:
            s_list.pop(0)
            y_list.pop(0)
            rho_list.pop(0)

        if skyk > 1e-10 * np.linalg.norm(sk) * np.linalg.norm(yk):
            s_list.append(sk.copy())
            y_list.append(yk.copy())
            rho_list.append(1.0 / skyk)

        x = x + sk
        fk = fk_new
        gk = gk_new

    return {"x": x, "f_val": fk, "grad_norm": np.linalg.norm(gk),
            "iterations": max_iter, "converged": False, "history": history}


def _lbfgs_direction(gk: np.ndarray, s_list: list, y_list: list,
                     rho_list: list, m: int) -> np.ndarray:
    """L-BFGS 两循环递归 (Nocedal & Wright Algorithm 7.4)."""
    q = gk.copy()
    n_hist = len(s_list)
    alpha_hist = np.zeros(n_hist)

    # 第一循环 (从新到旧)
    for i in range(n_hist - 1, -1, -1):
        alpha_hist[i] = rho_list[i] * np.dot(s_list[i], q)
        q = q - alpha_hist[i] * y_list[i]

    # 初始 Hessian 缩放 (Oren-Luenberger)
    if n_hist > 0:
        skyk = np.dot(s_list[-1], y_list[-1])
        ykyk = np.dot(y_list[-1], y_list[-1])
        gamma = skyk / max(ykyk, 1e-300)
        r = gamma * q
    else:
        r = q.copy()

    # 第二循环 (从旧到新)
    for i in range(n_hist):
        beta = rho_list[i] * np.dot(y_list[i], r)
        r = r + s_list[i] * (alpha_hist[i] - beta)

    return -r


# ---------------------------------------------------------------------------
# 5. 梯度下降法 (带自适应步长)
# ---------------------------------------------------------------------------

def gradient_descent(f: Callable, grad_f: Callable, x0: np.ndarray,
                     tol: float = 1e-10, max_iter: int = 5000,
                     lr_init: float = 0.01) -> Dict:
    """梯度下降法 (最速下降法).
    x_{k+1} = x_k - α_k ∇f(x_k)

    收敛速度: 线性, 收敛率 (κ-1)/(κ+1), κ 为条件数.
    对病态问题极慢 (锯齿现象).
    """
    x = x0.copy().astype(float)
    fk = f(x)
    gk = grad_f(x)
    history = []

    for k in range(max_iter):
        gnorm = np.linalg.norm(gk)
        history.append({"iter": k, "f": fk, "gnorm": gnorm})

        if gnorm < tol:
            return {"x": x, "f_val": fk, "grad_norm": gnorm,
                    "iterations": k, "converged": True, "history": history}

        direction = -gk
        alpha, fk = armijo_backtrack(f, x, gk, direction, alpha_init=lr_init, f_val=fk)
        x = x + alpha * direction
        gk = grad_f(x)

    return {"x": x, "f_val": fk, "grad_norm": np.linalg.norm(gk),
            "iterations": max_iter, "converged": False, "history": history}


# ---------------------------------------------------------------------------
# 6. 余弦相似度目标 (源自 1286_Shukti042)
# ---------------------------------------------------------------------------

def cosine_similarity_loss(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度损失: L = 1 - cos(a,b) = 1 - a·b/(||a||·||b||).
    源自 1286_Shukti042/benchmark_training.py 的 cosine_sim_1d.
    用于特征匹配优化.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-300 or nb < 1e-300:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)


def cosine_similarity_loss_grad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """余弦相似度损失关于 a 的梯度.
    ∇_a L = -b/(||a||·||b||) + (a·b)/(||a||³·||b||) · a
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-300 or nb < 1e-300:
        return np.zeros_like(a)
    ab = np.dot(a, b)
    return -b / (na * nb) + ab * a / (na ** 3 * nb)


# ---------------------------------------------------------------------------
# 7. 收敛诊断
# ---------------------------------------------------------------------------

def check_convergence(history: list, tol: float = 1e-10) -> Dict:
    """收敛诊断 (源自 572_ill_bvp 的收敛判断).
    判断标准:
      1. 梯度范数 < tol
      2. 函数值变化 < tol
      3. 迭代步数是否达到上限
    """
    if len(history) < 2:
        return {"converged": False, "reason": "迭代不足"}

    last = history[-1]
    prev = history[-2]

    if last["gnorm"] < tol:
        return {"converged": True, "reason": "梯度收敛",
                "gnorm": last["gnorm"]}

    df = abs(last["f"] - prev["f"])
    if df < tol * max(1.0, abs(last["f"])):
        return {"converged": True, "reason": "函数值收敛",
                "df": df}

    return {"converged": False, "reason": "未收敛",
            "gnorm": last["gnorm"], "df": df}


def estimate_convergence_rate(history: list) -> float:
    """估计收敛阶 (通过连续误差比).
    线性收敛: ||e_{k+1}||/||e_k|| → r < 1
    超线性: ||e_{k+1}||/||e_k|| → 0
    二次: log(||e_{k+1}||)/log(||e_k||²) → 1

    简化估计: 用梯度范数代替误差.
    """
    if len(history) < 3:
        return 0.0

    # 取最后 5 步
    recent = history[-5:] if len(history) >= 5 else history
    gnorms = [h["gnorm"] for h in recent if h["gnorm"] > 1e-300]

    if len(gnorms) < 2:
        return 0.0

    # 比值
    ratios = []
    for i in range(1, len(gnorms)):
        if gnorms[i - 1] > 1e-300:
            ratios.append(gnorms[i] / gnorms[i - 1])

    if len(ratios) == 0:
        return 0.0
    return float(np.mean(ratios))
