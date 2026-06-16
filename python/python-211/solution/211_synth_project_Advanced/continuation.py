"""
continuation.py
===============
延拓法 / 同伦法 —— 全局优化与路径跟踪

融合种子项目:
  - 210_continuation: 延拓步、Newton 校正、切线计算
  - 572_ill_bvp: 病态系统处理

核心公式:
  1. 同伦映射: H(x,t) = (1-t)G(x) + tF(x), t∈[0,1]
     G(x) 易解, F(x) 为目标; 从 t=0 跟踪到 t=1
  2. 预测步 (Euler): x₁ = x₀ + h·T₀, T₀ 为切向量
  3. 校正步 (Newton): 解 H(x₁,x₁(p)) = 0, 固定参数 p
  4. 切向量: T = null(∇_x H) (通过 SVD 或 QR)
  5. 步长自适应: h_new = h · √(tol/||Δx||)
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict


# ---------------------------------------------------------------------------
# 1. 切向量计算 (源自 210_continuation/tangent)
# ---------------------------------------------------------------------------

def compute_tangent(J: np.ndarray) -> np.ndarray:
    """计算延拓系统的切向量.
    给定 Jacobian J (n-1)×n, 切向量 T ∈ null(J).

    方法: SVD 最小奇异值对应的右奇异向量.
    J = U Σ V^T, T = V[:, -1] (对应最小奇异值).

    定向: 确保与前一切向量同向 (避免路径折返).
    """
    U, S, Vt = np.linalg.svd(J)
    tangent = Vt[-1, :]  # 最小奇异值对应的行

    # 归一化
    norm = np.linalg.norm(tangent)
    if norm > 1e-300:
        tangent /= norm
    return tangent


def compute_tangent_with_orientation(J: np.ndarray,
                                     prev_tangent: Optional[np.ndarray] = None) -> np.ndarray:
    """带定向的切向量计算.
    若提供了前一切向量, 确保新切向量与其同向:
      若 T·T_prev < 0, 则 T ← -T
    """
    tangent = compute_tangent(J)
    if prev_tangent is not None:
        if np.dot(tangent, prev_tangent) < 0:
            tangent = -tangent
    return tangent


# ---------------------------------------------------------------------------
# 2. Newton 校正步 (源自 210_continuation/newton)
# ---------------------------------------------------------------------------

def newton_corrector(F: Callable, JF: Callable, x0: np.ndarray,
                     param_idx: int, param_val: float,
                     tol: float = 1e-10, max_iter: int = 50) -> Tuple[int, np.ndarray]:
    """Newton 校正步 (源自 210_continuation/newton).

    解增广系统:
      G(x) = [ F(x)             ]   (n-1 个方程)
             [ x[param_idx] - v ]   (1 个方程, 固定参数)

    Jacobian:
      JG(x) = [ JF(x)        ]   (n-1)×n
              [ e_{param_idx} ]   1×n (第 param_idx 个标准基)

    Newton 步: x_{k+1} = x_k - JG^{-1} G(x_k)

    返回 (status, x): status=0 收敛, status=1 未收敛.
    """
    x = x0.copy()
    n = len(x)

    for iteration in range(max_iter):
        Fx = F(x)
        if not isinstance(Fx, np.ndarray):
            Fx = np.array([Fx])

        # 增广残差
        Gx = np.zeros(n)
        Gx[:n - 1] = Fx[:n - 1] if len(Fx) >= n - 1 else Fx
        Gx[n - 1] = x[param_idx] - param_val

        if np.linalg.norm(Gx) < tol:
            return 0, x

        # 增广 Jacobian
        JFx = JF(x)
        if JFx.ndim == 1:
            JFx = JFx.reshape(1, -1)
        JG = np.zeros((n, n))
        JG[:n - 1, :] = JFx[:n - 1, :] if JFx.shape[0] >= n - 1 else JFx
        JG[n - 1, param_idx] = 1.0

        try:
            delta = np.linalg.solve(JG, -Gx)
        except np.linalg.LinAlgError:
            # 使用最小二乘
            delta, _, _, _ = np.linalg.lstsq(JG, -Gx, rcond=None)

        x = x + delta

    return 1, x


# ---------------------------------------------------------------------------
# 3. 延拓步 (源自 210_continuation/step)
# ---------------------------------------------------------------------------

def continuation_step(F: Callable, JF: Callable,
                      x0: np.ndarray, t0: np.ndarray, p0: int,
                      h: float, tol: float = 1e-10) -> Tuple[int, np.ndarray, np.ndarray, int]:
    """延拓步 (源自 210_continuation/step).

    给定当前解 x0, 切向量 t0, 参数索引 p0, 步长 h:
      1. 预测: x1 = x0 + h · t0
      2. 校正: Newton 法解 H(x) = 0, 固定 x[p0] = x1[p0]

    返回 (status, x_new, t_new, p_new).
    """
    # 预测步
    x1 = x0 + h * t0

    # 新参数值
    param_val = x1[p0]

    # 校正步
    status, x2 = newton_corrector(F, JF, x1, p0, param_val, tol)

    # 新切向量
    JFx2 = JF(x2)
    if JFx2.ndim == 1:
        JFx2 = JFx2.reshape(1, -1)
    t2 = compute_tangent_with_orientation(JFx2, t0)

    # 自适应参数选择: 选择切向量中绝对值最大的分量
    p2 = int(np.argmax(np.abs(t2)))

    return status, x2, t2, p2


# ---------------------------------------------------------------------------
# 4. 自然参数延拓 (用于一维参数族)
# ---------------------------------------------------------------------------

def natural_parameter_continuation(F: Callable, JF: Callable,
                                   x0: np.ndarray,
                                   param_range: Tuple[float, float],
                                   n_steps: int = 50,
                                   tol: float = 1e-10) -> List[Dict]:
    """自然参数延拓.

    给定参数化系统 F(x, λ) = 0, λ ∈ [λ_min, λ_max]:
      从 λ=λ_min 的已知解 x0 开始, 逐步增加 λ.

    每步:
      1. 预测: x_pred = x_k
      2. 校正: 解 F(x, λ_{k+1}) = 0

    返回解路径 [{lambda, x, status}, ...].
    """
    lam_min, lam_max = param_range
    dlam = (lam_max - lam_min) / n_steps
    path = []

    x = x0.copy()
    lam = lam_min

    for step in range(n_steps + 1):
        # 校正步 (固定 λ)
        def F_lam(xx):
            return F(xx, lam)

        def JF_lam(xx):
            return JF(xx, lam)

        status, x_new = newton_corrector(F_lam, JF_lam, x,
                                         param_idx=0, param_val=0.0,
                                         tol=tol)

        residual = np.linalg.norm(F(x_new, lam))
        path.append({
            "step": step,
            "lambda": lam,
            "x": x_new.copy(),
            "residual": residual,
            "converged": residual < tol
        })

        x = x_new
        lam += dlam

    return path


# ---------------------------------------------------------------------------
# 5. 伪弧长延拓 (处理折返/分岔)
# ---------------------------------------------------------------------------

def pseudo_arclength_continuation(F: Callable, JF: Callable,
                                  x0: np.ndarray,
                                  h: float = 0.1,
                                  n_steps: int = 100,
                                  tol: float = 1e-10) -> List[Dict]:
    """伪弧长延拓 (处理极限点和折返).

    与延拓不同, 弧长参数化可以跟踪折返:
      增广系统: [F(x)] = 0, 附加 (x-x0)^T t0 + (s-s0) - ds = 0
                其中 s 为弧长参数, t0 为切向量.

    算法:
      1. 切向量: t = null(JF)
      2. 预测: (x_pred, s_pred) = (x + h·t, s + h)
      3. 校正: 解增广 Newton 系统
    """
    path = []
    x = x0.copy()
    s = 0.0

    # 初始切向量
    JF0 = JF(x)
    if JF0.ndim == 1:
        JF0 = JF0.reshape(-1, 1)
    t = compute_tangent(JF0)
    p = int(np.argmax(np.abs(t)))

    for step in range(n_steps):
        status, x_new, t_new, p = continuation_step(
            lambda xx: F(xx), lambda xx: JF(xx),
            x, t, p, h, tol
        )

        s += h
        residual = np.linalg.norm(F(x_new))
        path.append({
            "step": step,
            "s": s,
            "x": x_new.copy(),
            "tangent": t_new.copy(),
            "residual": residual,
            "converged": status == 0
        })

        x = x_new
        t = t_new

        # 自适应步长
        if status == 0 and residual < tol * 10:
            h *= 1.1  # 稍微增大
        elif status != 0:
            h *= 0.5  # 缩小步长

        h = max(min(h, 1.0), 1e-4)

    return path


# ---------------------------------------------------------------------------
# 6. 同伦法全局优化
# ---------------------------------------------------------------------------

def homotopy_optimization(f_target: Callable, grad_target: Callable,
                          f_start: Callable, grad_start: Callable,
                          x0: np.ndarray,
                          n_stages: int = 20,
                          tol: float = 1e-8,
                          optimizer=None) -> Dict:
    """同伦法全局优化.

    构造同伦: H(x,t) = (1-t)·f_start(x) + t·f_target(x)
    从 t=0 (易解) 逐步跟踪到 t=1 (目标).

    每阶段:
      1. 优化 H(x, t_k) 从上一阶段的解出发
      2. t_{k+1} = t_k + Δt

    这使优化器能穿越能量壁垒, 找到更好的极小点.

    返回 dict: {x, f_val, path, converged}.
    """
    if optimizer is None:
        from quasi_newton import bfgs
        optimizer = bfgs

    x = x0.copy()
    path = []
    t_values = np.linspace(0, 1, n_stages + 1)

    for k, t in enumerate(t_values):
        # 构造同伦目标
        def f_homotopy(xx, t=t):
            return (1.0 - t) * f_start(xx) + t * f_target(xx)

        def grad_homotopy(xx, t=t):
            return (1.0 - t) * grad_start(xx) + t * grad_target(xx)

        # 优化当前同伦阶段
        result = optimizer(f_homotopy, grad_homotopy, x, tol=tol, max_iter=200)
        x = result["x"]

        path.append({
            "stage": k,
            "t": t,
            "f_target": f_target(x),
            "f_homotopy": f_homotopy(x),
            "gnorm": result["grad_norm"]
        })

    return {
        "x": x,
        "f_val": f_target(x),
        "path": path,
        "converged": path[-1]["gnorm"] < tol if path else False
    }
