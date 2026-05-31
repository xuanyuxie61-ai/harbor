r"""
continuation_solver.py
=====================
伪弧长延拓法（Pseudo-Arclength Continuation）求解器，
用于追踪太阳耀斑磁重联平衡态随反常电阻率参数 eta 变化的解分支。

核心数学原理
------------
给定非线性方程组 F(U, eta) = 0，其中 U 为状态变量，eta 为延续参数。
延拓法通过引入弧长参数 s，求解增广系统:

    G(U, eta) = | F(U, eta)            | = 0
                 | N(U, eta, s)         |

其中伪弧长约束 N 为:
    N = (U - U_0)^T * dU/ds + (eta - eta_0) * d eta/ds - Delta s = 0

切向量 T = (dU/ds, d eta/ds)^T 满足:
    [ F_U   F_eta ] [ dU/ds   ]   [ 0 ]
    [ dU_0  d eta_0] [ d eta/ds] = [ 1 ]

Newton 迭代步骤:
    [ F_U   F_eta ] [ delta U   ]   [ -F(U, eta) ]
    [ dU_0  d eta_0] [ delta eta] = [ -N        ]

步长自适应控制:
    h_{new} = h_{old} * sqrt( N_{target} / N_{actual} )

融入原项目:
- 210_continuation: Newton 迭代、切向量计算、步进控制
"""

import numpy as np
from typing import Callable, Tuple, List, Optional


class ContinuationSolver:
    """
    伪弧长延拓法求解器，用于追踪磁重联平衡态解曲线。
    """

    def __init__(self,
                 max_iter: int = 20,
                 tol: float = 1e-10,
                 h_init: float = 0.05,
                 h_min: float = 1e-6,
                 h_max: float = 0.5,
                 verbose: bool = False):
        self.max_iter = max_iter
        self.tol = tol
        self.h = h_init
        self.h_min = h_min
        self.h_max = h_max
        self.verbose = verbose

    def _newton_step(self,
                     F: Callable[[np.ndarray], np.ndarray],
                     J: Callable[[np.ndarray], np.ndarray],
                     x0: np.ndarray,
                     p: int) -> Tuple[int, np.ndarray]:
        """
        对增广系统执行 Newton 迭代，固定第 p 个分量。

        增广系统:
            G(x) = [ F(x)       ]
                   [ x[p] - a   ]

        Jacobian:
            JG = [ JF(x)      ]
                 [ e_p^T      ]
        """
        x = np.copy(x0)
        alpha = x0[p]
        n = len(x)

        for it in range(self.max_iter):
            fx = F(x)
            fx = np.append(fx, x[p] - alpha)
            fx_norm = np.max(np.abs(fx))

            if self.verbose:
                print(f"    Newton iter {it}: ||F||_inf = {fx_norm:.3e}")

            if fx_norm <= self.tol:
                return 0, x

            it += 1

            jf = J(x)
            # 增广 Jacobian
            jg = np.zeros((n, n))
            jg[:n-1, :] = jf
            jg[n-1, :] = 0.0
            jg[n-1, p] = 1.0

            try:
                dx = -np.linalg.solve(jg, fx)
            except np.linalg.LinAlgError:
                return 1, x

            x = x + dx

        return 1, x

    def _compute_tangent(self,
                         J: Callable[[np.ndarray], np.ndarray],
                         x: np.ndarray,
                         t_prev: np.ndarray,
                         p: int) -> Tuple[np.ndarray, int]:
        """
        计算单位切向量 t，并选择下一个延续参数索引 p2。

        解线性系统:
            [ JF(x)      ] t = [ 0 ]
            [ e_p^T      ]     [ 1 ]

        归一化: t = t / ||t||
        保持方向一致性: t^T * t_prev > 0
        """
        n = len(x)
        jf = J(x)
        jg = np.zeros((n, n))
        jg[:n-1, :] = jf
        jg[n-1, :] = 0.0
        jg[n-1, p] = 1.0

        b = np.zeros(n)
        b[n-1] = 1.0

        try:
            t = np.linalg.solve(jg, b)
        except np.linalg.LinAlgError:
            # 如果奇异，使用最小二乘
            t = np.linalg.lstsq(jg, b, rcond=None)[0]

        t_norm = np.linalg.norm(t)
        if t_norm < 1e-14:
            raise RuntimeError("切向量范数接近零，可能到达分歧点")
        t = t / t_norm

        # 方向一致性
        if np.dot(t, t_prev) < 0.0:
            t = -t

        # 选择绝对值最大的分量作为下一个延续参数
        p2 = int(np.argmax(np.abs(t)))
        return t, p2

    def trace_branch(self,
                     F: Callable[[np.ndarray], np.ndarray],
                     J: Callable[[np.ndarray], np.ndarray],
                     x0: np.ndarray,
                     param_index: int = -1,
                     n_steps: int = 50) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        追踪解分支。

        Parameters
        ----------
        F : callable
            非线性方程组 F(x) = 0，x 的最后分量为参数 eta。
        J : callable
            Jacobian 矩阵 JF(x)。
        x0 : ndarray
            初始解点（包含状态变量和参数）。
        param_index : int
            初始延续参数在 x 中的索引，默认最后一个分量。
        n_steps : int
            延拓步数。

        Returns
        -------
        xs : list of ndarray
            解序列。
        params : list of float
            参数序列。
        ps : list of int
            每步使用的延续参数索引。
        """
        x = np.array(x0, dtype=float, copy=True)
        n = len(x)
        if param_index < 0:
            param_index = n - 1

        t_prev = np.zeros(n)
        t_prev[param_index] = 1.0

        xs = [np.copy(x)]
        params = [float(x[param_index])]
        ps = [param_index]

        for step in range(n_steps):
            # 计算切向量
            try:
                t, p_next = self._compute_tangent(J, x, t_prev, param_index)
            except RuntimeError as e:
                if self.verbose:
                    print(f"[Continuation] 步 {step}: {e}")
                break

            # 预测步: x_pred = x + h * t
            x_pred = x + self.h * t

            # 校正步（固定参数 p_next）
            status, x_new = self._newton_step(F, J, x_pred, p_next)

            if status != 0:
                # 步长过大，缩小重试
                self.h *= 0.5
                if self.h < self.h_min:
                    if self.verbose:
                        print(f"[Continuation] 步 {step}: 步长小于最小值，终止")
                    break
                continue

            # 成功，更新
            x = x_new
            param_index = p_next
            t_prev = t

            xs.append(np.copy(x))
            params.append(float(x[param_index]))
            ps.append(param_index)

            # 自适应步长调整（简单策略：成功则略微增大）
            self.h = min(self.h * 1.1, self.h_max)

            if self.verbose:
                print(f"[Continuation] 步 {step}: param={x[param_index]:.5e}, ||F||收敛")

        return xs, params, ps


def demo_mhd_continuation():
    """
    演示：追踪简化 MHD 平衡态中参数 eta（电阻率）变化时的解分支。
    模型方程（单自由度非线性弹簧-磁压平衡）:
        F1(u, eta) = u^3 - u + eta = 0
    这是一个具有 fold 分歧结构的典型方程。
    """
    print("\n[ContinuationSolver] 演示: 追踪 fold 分歧")

    def F(x):
        u = x[0]
        eta = x[1]
        return np.array([u**3 - u + eta])

    def J(x):
        u = x[0]
        return np.array([[3.0 * u**2 - 1.0, 1.0]])

    # 初始解: u=0, eta=0
    x0 = np.array([0.0, 0.0])
    solver = ContinuationSolver(h_init=0.05, tol=1e-12, verbose=False)
    xs, params, ps = solver.trace_branch(F, J, x0, n_steps=80)

    print(f"  成功追踪 {len(xs)} 个点")
    print(f"  参数范围: eta = [{min(params):.4f}, {max(params):.4f}]")
    # 检查是否捕获到 fold 点（导数 du/deta -> inf 对应 3u^2-1=0）
    fold_candidates = [u for u, _ in xs if abs(3.0 * u**2 - 1.0) < 0.1]
    if fold_candidates:
        print(f"  检测到 fold 分歧点附近: u ~ {np.mean(fold_candidates):.4f}")
    return xs, params


if __name__ == "__main__":
    demo_mhd_continuation()
