"""
隐式辐射扩散求解器 (from 760_mgmres GMRES + 965_r83s 稀疏带状存储).

辐射扩散方程 (Flux-limited diffusion, FLD):
  ∂E_r/∂t = ∇·(D ∇ E_r) - χ_a c E_r + χ_a c a T^4

其中 E_r = 辐射能密度, χ_a = 吸收系数, D = λ c / χ_t 为扩散系数,
λ 为 flux limiter, χ_t = χ_a + χ_s 为总消光系数.

离散化：
  (E_r^{n+1} - E_r^n) / Δt = L(E_r^{n+1})
  (I - Δt L) E_r^{n+1} = E_r^n

得到稀疏线性系统 A x = b, 其中 A = I - Δt L 为五对角带矩阵
(一维球对称，每单元耦合左右相邻单元).

采用**稀疏带状压缩存储** (from 965_r83s):
  - 对角元存入 a[0..N-1]
  - 上次对角 (sub-diagonal) a_sub[0..N-2]
  - 下次对角 (super-diagonal) a_sup[0..N-2]
  - 辐射扩散在五对角时需要两条额外的次对角

求解使用**重启 GMRES(m)** (from 760_mgmres):
  给定初始猜测 x_0, 计算 r_0 = b - A x_0,
  构造 Krylov 子空间 K_m(A, r_0) = span{r_0, A r_0, ..., A^{m-1} r_0},
  通过 Arnoldi 过程得到 Hessenberg 矩阵 H_m,
  在 K_m 中求 min ||b - A x|| 的最小二乘解.
  重启参数 m = 20 (典型).

收敛判定：
  ||r_k|| / ||r_0|| < tol_rel 且 ||r_k|| < tol_abs.

预处理：ILU(0) 不完全 LU 分解 (from 760_mgmres ilu_crs).
"""
from __future__ import annotations
import math
import numpy as np


class BandMatrix:
    """三对角/五对角稀疏带状矩阵 (from 965_r83s r83 存储格式).

    存储 (对 N 阶方阵):
      d : (N,) 主对角
      dl : (N-1,) 下次对角 (sub-diagonal)
      du : (N-1,) 上次对角 (super-diagonal)
      dl2 : (N-2,) 下次次对角 (仅五对角)
      du2 : (N-2,) 上次次对角 (仅五对角)
    """

    def __init__(self, N: int, kind: str = 'tridiag'):
        self.N = int(N)
        self.kind = kind
        self.d = np.zeros(N, dtype=np.float64)
        if kind in ('tridiag', 'pentadiag'):
            self.dl = np.zeros(N - 1, dtype=np.float64)
            self.du = np.zeros(N - 1, dtype=np.float64)
        if kind == 'pentadiag':
            self.dl2 = np.zeros(N - 2, dtype=np.float64)
            self.du2 = np.zeros(N - 2, dtype=np.float64)

    def mv(self, x: np.ndarray) -> np.ndarray:
        """矩阵-向量乘法 y = A x."""
        N = self.N
        y = self.d * x
        if self.kind in ('tridiag', 'pentadiag'):
            y[:-1] += self.du * x[1:]
            y[1:] += self.dl * x[:-1]
        if self.kind == 'pentadiag':
            y[:-2] += self.du2 * x[2:]
            y[2:] += self.dl2 * x[:-2]
        return y

    def tridiag_solve(self, b: np.ndarray) -> np.ndarray:
        """Thomas 算法解三对角系统 (O(N) 时间)."""
        if self.kind != 'tridiag':
            raise ValueError("仅适用于三对角")
        N = self.N
        c = np.zeros(N, dtype=np.float64)
        d = np.zeros(N, dtype=np.float64)
        x = np.zeros(N, dtype=np.float64)
        c[0] = self.du[0] / self.d[0]
        d[0] = b[0] / self.d[0]
        for i in range(1, N - 1):
            m = self.d[i] - self.dl[i - 1] * c[i - 1]
            if abs(m) < 1.0e-30:
                m = 1.0e-30
            c[i] = self.du[i] / m
            d[i] = (b[i] - self.dl[i - 1] * d[i - 1]) / m
        m = self.d[N - 1] - self.dl[N - 2] * c[N - 2]
        if abs(m) < 1.0e-30:
            m = 1.0e-30
        d[N - 1] = (b[N - 1] - self.dl[N - 2] * d[N - 2]) / m
        x[N - 1] = d[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]
        return x


class RestartedGMRES:
    """重启 GMRES(m) 求解 A x = b (from 760_mgmres).

    Arnoldi 过程生成 Hessenberg 矩阵 H, 每 m 步重启.
    使用 Givens 旋转 (from 760_mgmres mult_givens) 解最小二乘.
    """

    def __init__(self, max_iter: int = 100, restart: int = 20,
                 tol_abs: float = 1.0e-10, tol_rel: float = 1.0e-10):
        self.max_iter = int(max_iter)
        self.restart = int(restart)
        self.tol_abs = float(tol_abs)
        self.tol_rel = float(tol_rel)
        self.history_residual = []
        self.total_iters = 0

    def _givens_rotation(self, a: float, b: float):
        """Givens 旋转参数 (c, s), 使 (c s; -s c)(a; b) = (r; 0)."""
        if abs(b) < 1.0e-30:
            return 1.0, 0.0
        elif abs(a) < 1.0e-30:
            return 0.0, 1.0 if b > 0 else -1.0
        r = math.hypot(a, b)
        return a / r, b / r

    def solve(self, A, b: np.ndarray, x0: np.ndarray | None = None,
              precond=None) -> tuple[np.ndarray, dict]:
        """求解 A x = b. A 为可调用 mv(x) 的对象或函数."""
        N = len(b)
        if x0 is None:
            x = np.zeros(N, dtype=np.float64)
        else:
            x = x0.copy()
        r = b - A.mv(x)
        if precond is not None:
            r = precond(r)
        beta = float(np.linalg.norm(r))
        self.history_residual = [beta]
        b_norm = max(float(np.linalg.norm(b)), 1.0e-30)
        total = 0
        m = self.restart
        while total < self.max_iter and beta / b_norm > self.tol_rel and beta > self.tol_abs:
            # Arnoldi
            V = np.zeros((m + 1, N), dtype=np.float64)
            H = np.zeros((m + 1, m), dtype=np.float64)
            V[0] = r / max(beta, 1.0e-30)
            # Givens 旋转存储
            cs = np.zeros(m, dtype=np.float64)
            sn = np.zeros(m, dtype=np.float64)
            e1 = np.zeros(m + 1, dtype=np.float64)
            e1[0] = beta
            converged = False
            j_end = m
            for j in range(m):
                w = A.mv(V[j])
                if precond is not None:
                    w = precond(w)
                for i in range(j + 1):
                    H[i, j] = float(np.dot(w, V[i]))
                    w = w - H[i, j] * V[i]
                H[j + 1, j] = float(np.linalg.norm(w))
                if H[j + 1, j] < 1.0e-30:
                    # 幸运中断
                    j_end = j + 1
                    converged = True
                    break
                V[j + 1] = w / H[j + 1, j]
                # 对 H 列 j 应用先前 Givens 旋转
                for i in range(j):
                    temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                    H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                    H[i, j] = temp
                cs[j], sn[j] = self._givens_rotation(H[j, j], H[j + 1, j])
                H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
                H[j + 1, j] = 0.0
                e1[j + 1] = -sn[j] * e1[j]
                e1[j] = cs[j] * e1[j]
                res = abs(e1[j + 1])
                self.history_residual.append(res)
                total += 1
                if res / b_norm < self.tol_rel or res < self.tol_abs:
                    j_end = j + 1
                    converged = True
                    break
            # 回代 H y = e1
            y = np.zeros(j_end, dtype=np.float64)
            for i in range(j_end - 1, -1, -1):
                y[i] = e1[i]
                for k in range(i + 1, j_end):
                    y[i] -= H[i, k] * y[k]
                if abs(H[i, i]) < 1.0e-30:
                    y[i] = 0.0
                else:
                    y[i] /= H[i, i]
            # 更新 x
            for j in range(j_end):
                x = x + y[j] * V[j]
            # 计算新残差
            r = b - A.mv(x)
            if precond is not None:
                r = precond(r)
            beta = float(np.linalg.norm(r))
            self.history_residual.append(beta)
            if converged and (beta / b_norm < self.tol_rel or beta < self.tol_abs):
                break
        self.total_iters = total
        info = {'converged': beta / b_norm < self.tol_rel or beta < self.tol_abs,
                'final_residual': beta,
                'iters': total,
                'history': self.history_residual}
        return x, info


def ilu0_preconditioner(A: BandMatrix):
    """ILU(0) 不完全 LU 分解预条件子.

    对三对角 A = (dl, d, du), ILU(0) 给出:
      L = (1, dl_new),  U = (d_new, du)
    其中 d_new[0] = d[0], d_new[i] = d[i] - dl[i-1] du[i-1] / d_new[i-1].
    """
    N = A.N
    d_new = np.zeros(N, dtype=np.float64)
    dl_new = np.zeros(N - 1, dtype=np.float64)
    du = A.du.copy() if hasattr(A, 'du') else np.zeros(N - 1)
    d_new[0] = A.d[0]
    for i in range(1, N):
        if abs(d_new[i - 1]) < 1.0e-30:
            d_new[i - 1] = 1.0e-30
        dl_new[i - 1] = A.dl[i - 1] / d_new[i - 1]
        d_new[i] = A.d[i] - dl_new[i - 1] * du[i - 1]
        if abs(d_new[i]) < 1.0e-30:
            d_new[i] = 1.0e-30

    def precond(r):
        # 解 L y = r
        y = np.zeros(N, dtype=np.float64)
        y[0] = r[0]
        for i in range(1, N):
            y[i] = r[i] - dl_new[i - 1] * y[i - 1]
        # 解 U x = y
        x = np.zeros(N, dtype=np.float64)
        x[N - 1] = y[N - 1] / d_new[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = (y[i] - du[i] * x[i + 1]) / d_new[i]
        return x

    return precond
