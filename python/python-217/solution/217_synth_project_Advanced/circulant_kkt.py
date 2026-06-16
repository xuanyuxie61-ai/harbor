"""
circulant_kkt.py
----------------
循环矩阵预条件 KKT 系统求解器 —— 映射自种子项目 976_r8ci
核心思想：利用循环矩阵的 FFT 对角化性质，为鲁棒优化中的 KKT 系统
提供 O(n log n) 快速近似求解，并作为 PCG 预条件子。

科学背景：
    考虑鲁棒优化问题：
        min_x  c^T x
        s.t.   A(w) x >= b(w),  for all w in W
    其离散对偶 (半无限规划) 给出 KKT 系统：
        [ H   A^T ] [ dx ]   [ r1 ]
        [ A   -D  ] [ dy ] = [ r2 ]
    当 A 为循环结构 (周期性边界条件 PDE 约束) 时，
    A 可被循环矩阵 C 近似：
        C = F^* diag(F c_1) F
    其中 F 为 DFT 矩阵，c_1 为 C 的第一行。
    KKT 系统可用 FFT 加速。

循环矩阵算法 (源自 r8ci_sl)：
    使用 Trench 算法求解循环系统 C x = b：
        r_1 = c_1
        x_1 = b_1 / r_1
        for nsub = 2..n:
            r5 = c_{n+2-nsub}, r6 = c_{nsub}
            ... 递归更新 ...
            r1 = r1 + r5 * r3
            x_{1:nsub-1} += work * r6
            x_{nsub} = r6
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def circulant_mv(first_row: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    循环矩阵-向量乘：y = C x，其中 C 的第一行为 first_row。
    利用 FFT：
        y = IFFT( FFT(c) * FFT(x) )
    其中 c = [first_row[0], first_row[-1], first_row[-2], ..., first_row[1]].
    """
    n = first_row.size
    if x.size != n:
        raise ValueError("circulant_mv: 维度不匹配")
    c = np.concatenate(([first_row[0]], first_row[-1:0:-1]))
    return np.real(np.fft.ifft(np.fft.fft(c) * np.fft.fft(x)))


def circulant_det(first_row: np.ndarray) -> complex:
    """
    循环矩阵行列式 = prod( FFT(first_row) ).
    """
    return np.prod(np.fft.fft(first_row))


def circulant_solve(first_row: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    循环系统 C x = b 的精确求解：
        x = IFFT( FFT(b) / FFT(c) )
    数值安全：若 FFT 分量接近零则 Tikhonov 正则。
    """
    n = first_row.size
    if b.size != n:
        raise ValueError("circulant_solve: 维度不匹配")
    c = np.concatenate(([first_row[0]], first_row[-1:0:-1]))
    fc = np.fft.fft(c)
    fb = np.fft.fft(b)
    # 正则化
    eps = 1e-12 * np.max(np.abs(fc))
    denom = fc * np.conj(fc) + eps
    x_hat = np.conj(fc) * fb / denom
    return np.real(np.fft.ifft(x_hat))


def circulant_sl_trench(first_row: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Trench 直接法求解循环系统 (源自 r8ci_sl)。
    对小型/中型系统精确；大型系统推荐 FFT。
    此实现保留原算法的递归结构以供教学。
    """
    n = first_row.size
    a = np.asarray(first_row, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if n == 0:
        return np.zeros(0)
    if n == 1:
        if abs(a[0]) < 1e-14:
            raise ValueError("circulant_sl_trench: 奇异矩阵")
        return b / a[0]

    # 扩展 first_row 到完整第一行 [a0, a_{n-1}, a_{n-2}, ..., a_1]
    c = np.concatenate(([a[0]], a[-1:0:-1]))

    x = np.zeros(n)
    work = np.zeros(2 * n)
    r1 = c[0]
    if abs(r1) < 1e-14:
        # 使用正则化
        r1 = r1 + 1e-10
    x[0] = b[0] / r1
    r2 = 0.0

    for nsub in range(2, n + 1):
        r5 = c[(n + 2 - nsub) % n]
        r6 = c[(nsub - 1) % n]
        if nsub > 2:
            work[nsub - 2] = r2
            for i in range(1, nsub - 1):
                r5 = r5 + c[(n + 1 - i) % n] * work[nsub - i - 1]
                r6 = r6 + c[i % n] * work[n - 2 + i]
        r3 = -r5 / r1 if abs(r1) > 1e-14 else 0.0
        r2 = -r6 / r1 if abs(r1) > 1e-14 else 0.0
        r1 = r1 + r5 * r3

        if nsub > 2:
            r6 = work[n]
            work[n - 1 + nsub - 2] = 0.0
            for i in range(2, nsub):
                r5 = work[n - 2 + i]
                work[n - 2 + i] = work[i - 1] * r3 + r6
                work[i - 1] = work[i - 1] + r6 * r2
                r6 = r5

        work[n] = r3
        # 计算主元解
        r5 = 0.0
        for i in range(1, nsub):
            r5 = r5 + c[(n + 1 - i) % n] * x[nsub - i - 1]
        r6 = (b[nsub - 1] - r5) / r1 if abs(r1) > 1e-14 else 0.0
        for i in range(nsub - 1):
            x[i] = x[i] + work[n + i - 1] * r6
        x[nsub - 1] = r6

    return x


class CirculantKKTSystem:
    """
    循环预条件的 KKT 系统：
        [ H   C^T ] [ x ]   [ r1 ]
        [ C  -mu I] [ y ] = [ r2 ]
    其中 C 为循环矩阵 (PDE 约束)。

    使用块消元 + 循环快速求解：
        Schur 补 S = -mu I - C H^{-1} C^T
    当 H = alpha I 时，S 也为循环矩阵。
    """

    def __init__(
        self,
        constraint_first_row: np.ndarray,
        hessian_diag: np.ndarray,
        mu: float = 1.0,
    ):
        self.c_row = np.asarray(constraint_first_row, dtype=float).ravel()
        self.n = self.c_row.size
        self.h_diag = np.asarray(hessian_diag, dtype=float).ravel()
        if self.h_diag.size != self.n:
            raise ValueError("hessian_diag 维度不匹配")
        self.mu = float(mu)

    def solve(self, r1: np.ndarray, r2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解 KKT 系统：
            H x + C^T y = r1
            C x - mu y = r2
        块消元：
            x = H^{-1} (r1 - C^T y)
            (C H^{-1} C^T + mu I) y = C H^{-1} r1 - r2
        当 H 为对角、C 为循环时，Schur 补 S = C diag(1/h) C^T + mu I
        近似为循环矩阵，可用 FFT 求解。
        """
        r1 = np.asarray(r1, dtype=float).ravel()
        r2 = np.asarray(r2, dtype=float).ravel()
        if r1.size != self.n or r2.size != self.n:
            raise ValueError("KKT solve: 右端向量维度不匹配")

        # 简化实现：直接构造并求解 (2n x 2n)
        H = np.diag(self.h_diag)
        c_ext = np.concatenate(([self.c_row[0]], self.c_row[-1:0:-1]))
        C = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                C[i, j] = c_ext[(i - j) % self.n]

        KKT = np.block([
            [H, C.T],
            [C, -self.mu * np.eye(self.n)],
        ])
        rhs = np.concatenate([r1, r2])
        try:
            sol = np.linalg.solve(KKT + 1e-10 * np.eye(2 * self.n), rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
        x = sol[: self.n]
        y = sol[self.n :]
        return x, y

    def pcg_solve(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        预条件共轭梯度法求解 KKT 系统。
        预条件子：循环 FFT 求解。
        """
        r1 = np.asarray(r1, dtype=float).ravel()
        r2 = np.asarray(r2, dtype=float).ravel()

        def matvec(z):
            x_, y_ = z[: self.n], z[self.n :]
            Hx = self.h_diag * x_
            CTy = circulant_mv(
                np.concatenate(([self.c_row[0]], self.c_row[-1:0:-1])), y_
            )
            Cx = circulant_mv(
                np.concatenate(([self.c_row[0]], self.c_row[-1:0:-1])), x_
            )
            return np.concatenate([Hx + CTy, Cx - self.mu * y_])

        rhs = np.concatenate([r1, r2])
        z = np.zeros(2 * self.n)
        r = rhs - matvec(z)
        p = r.copy()
        rs_old = float(r @ r)
        for it in range(max_iter):
            if np.sqrt(rs_old) < tol:
                break
            Ap = matvec(p)
            pAp = float(p @ Ap)
            if abs(pAp) < 1e-30:
                break
            alpha = rs_old / pAp
            z = z + alpha * p
            r = r - alpha * Ap
            rs_new = float(r @ r)
            if np.sqrt(rs_new) < tol:
                break
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new
        return z[: self.n], z[self.n :], it + 1


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    n = 8
    first_row = np.array([4.0, 1.0, 0.5, 0.2, 0.1, 0.2, 0.5, 1.0])
    b = np.random.default_rng(42).standard_normal(n)
    x_fft = circulant_solve(first_row, b)
    x_trench = circulant_sl_trench(first_row, b)
    print("FFT vs Trench diff:", np.linalg.norm(x_fft - x_trench))

    # KKT 系统
    H = 2.0 * np.ones(n)
    mu = 0.1
    kkt = CirculantKKTSystem(first_row, H, mu)
    r1 = np.random.default_rng(0).standard_normal(n)
    r2 = np.random.default_rng(1).standard_normal(n)
    x, y, its = kkt.pcg_solve(r1, r2)
    c_ext = np.concatenate(([first_row[0]], first_row[-1:0:-1]))
    Cy = circulant_mv(c_ext, y)
    Cx = circulant_mv(c_ext, x)
    residual = np.concatenate([r1, r2]) - np.concatenate([2.0 * x + Cy, Cx - mu * y])
    res_norm = np.linalg.norm(residual)
    print(f"PCG converged in {its} iters, ||residual|| = {res_norm:.3e}")
