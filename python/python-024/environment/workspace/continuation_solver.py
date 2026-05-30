
import numpy as np
from typing import Callable, Tuple, List, Optional


class ContinuationSolver:

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

            t = np.linalg.lstsq(jg, b, rcond=None)[0]

        t_norm = np.linalg.norm(t)
        if t_norm < 1e-14:
            raise RuntimeError("切向量范数接近零，可能到达分歧点")
        t = t / t_norm


        if np.dot(t, t_prev) < 0.0:
            t = -t


        p2 = int(np.argmax(np.abs(t)))
        return t, p2

    def trace_branch(self,
                     F: Callable[[np.ndarray], np.ndarray],
                     J: Callable[[np.ndarray], np.ndarray],
                     x0: np.ndarray,
                     param_index: int = -1,
                     n_steps: int = 50) -> Tuple[List[np.ndarray], List[float], List[int]]:
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

            try:
                t, p_next = self._compute_tangent(J, x, t_prev, param_index)
            except RuntimeError as e:
                if self.verbose:
                    print(f"[Continuation] 步 {step}: {e}")
                break


            x_pred = x + self.h * t


            status, x_new = self._newton_step(F, J, x_pred, p_next)

            if status != 0:

                self.h *= 0.5
                if self.h < self.h_min:
                    if self.verbose:
                        print(f"[Continuation] 步 {step}: 步长小于最小值，终止")
                    break
                continue


            x = x_new
            param_index = p_next
            t_prev = t

            xs.append(np.copy(x))
            params.append(float(x[param_index]))
            ps.append(param_index)


            self.h = min(self.h * 1.1, self.h_max)

            if self.verbose:
                print(f"[Continuation] 步 {step}: param={x[param_index]:.5e}, ||F||收敛")

        return xs, params, ps


def demo_mhd_continuation():
    print("\n[ContinuationSolver] 演示: 追踪 fold 分歧")

    def F(x):
        u = x[0]
        eta = x[1]
        return np.array([u**3 - u + eta])

    def J(x):
        u = x[0]
        return np.array([[3.0 * u**2 - 1.0, 1.0]])


    x0 = np.array([0.0, 0.0])
    solver = ContinuationSolver(h_init=0.05, tol=1e-12, verbose=False)
    xs, params, ps = solver.trace_branch(F, J, x0, n_steps=80)

    print(f"  成功追踪 {len(xs)} 个点")
    print(f"  参数范围: eta = [{min(params):.4f}, {max(params):.4f}]")

    fold_candidates = [u for u, _ in xs if abs(3.0 * u**2 - 1.0) < 0.1]
    if fold_candidates:
        print(f"  检测到 fold 分歧点附近: u ~ {np.mean(fold_candidates):.4f}")
    return xs, params


if __name__ == "__main__":
    demo_mhd_continuation()
