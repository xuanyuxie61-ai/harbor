"""
sparse_solver.py

DNA 损伤修复分子动力学 —— 稀疏/带状矩阵的隐式求解与 Poisson-Boltzmann 电静力学

基于种子项目:
  - 871_plasma_matrix: 等离子体问题的稀疏 Jacobian 与残差组装
  - 979_r8gb: 带状矩阵的 LU 分解 (LINPACK/LAPACK 风格)

科学背景:
  在 DNA 损伤修复的分子动力学模拟中，隐式时间积分需要求解大型稀疏
  线性系统；同时，修复蛋白与 DNA 骨架之间的静电相互作用可用
  Poisson-Boltzmann (PB) 方程描述:

      ∇ · (ε(r) ∇φ(r)) - κ^2 sinh(φ(r)) = -4π ρ(r) / ε_0

  在非线性 PB 方程的 Newton 迭代中，每一步需要求解形如 J δx = -F 的
  线性系统，其中 J 为 Jacobian（稀疏带状结构）。本模块实现了带状矩阵
  的 LU 分解 (r8gb_trf/r8gb_trs) 以及等离子体问题启发下的稀疏矩阵组装。
"""

import numpy as np
from typing import Tuple, Optional


class BandedLU:
    """
    带状矩阵的 PLU 分解与线性系统求解器。

    存储格式（R8GB/LAPACK 风格）:
        原始 n×n 带状矩阵 A（下半带宽 ml，上半带宽 mu）被"压缩"为
        (2*ml + mu + 1) × n 的二维数组，使得原矩阵的第 j 列的对角线
        元素出现在压缩数组的第 (ml + mu + 1) 行。

        具体映射:
            a_full(i, j)  ->  a_band(ml + mu + 1 + i - j, j)
        其中 max(1, j-mu) <= i <= min(n, j+ml)。
    """

    def __init__(self, n: int, ml: int, mu: int):
        self.n = n
        self.ml = ml
        self.mu = mu
        self.kv = mu + ml  # 分解后 U 的上带宽（含 fill-in）
        self.kd = mu + ml + 1

    def _full_to_band(self, A_full: np.ndarray) -> np.ndarray:
        """将稠密矩阵转换为带状存储。"""
        n = self.n
        ml = self.ml
        mu = self.mu
        rows = 2 * ml + mu + 1
        A_band = np.zeros((rows, n), dtype=np.float64)

        for j in range(n):
            i1 = max(0, j - mu)
            i2 = min(n - 1, j + ml)
            for i in range(i1, i2 + 1):
                k = i - j + ml + mu
                A_band[k, j] = A_full[i, j]
        return A_band

    def factorize(self, A_band: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        带状矩阵的 PLU 分解（简化版 LAPACK xGBTRF）。

        Returns
        -------
        A_lu : ndarray, shape (2*ml+mu+1, n)
            LU 因子存储在带状数组中。
        pivot : ndarray, shape (n,)
            行置换索引（1-based）。
        info : int
            0 表示成功；正值表示第 info 步遇到零主元。
        """
        n = self.n
        ml = self.ml
        mu = self.mu
        kv = self.kv

        A_lu = np.array(A_band, dtype=np.float64, copy=True)
        pivot = np.zeros(n, dtype=np.int64)
        info = 0

        # 初始化 fill-in 列为零
        for j in range(mu + 1, min(kv + 1, n)):
            for i in range(kv - j + 1, ml):
                A_lu[i, j] = 0.0

        ju = 0
        for j in range(min(n, n)):  # j = 0..n-1
            # 下一 fill-in 列置零
            if j + kv < n:
                A_lu[:ml, j + kv] = 0.0

            # 选主元
            km = min(ml, n - j - 1)
            piv = abs(A_lu[kv, j])
            jp = 0
            for i in range(1, km + 1):
                if abs(A_lu[kv + i, j]) > piv:
                    piv = abs(A_lu[kv + i, j])
                    jp = i

            pivot[j] = jp + j + 1  # 1-based

            if abs(A_lu[kv + jp, j]) < 1e-18:
                if info == 0:
                    info = j + 1
                continue

            # 更新受影响的最后一列
            ju = max(ju, min(j + mu + jp, n - 1))

            # 行交换
            if jp != 0:
                for i in range(ju - j + 1):
                    t = A_lu[kv + jp - i, j + i]
                    A_lu[kv + jp - i, j + i] = A_lu[kv - i, j + i]
                    A_lu[kv - i, j + i] = t

            # 计算乘子
            if km > 0:
                A_lu[kv + 1:kv + km + 1, j] /= A_lu[kv, j]

                # 更新尾随子矩阵
                if j < ju:
                    for k in range(1, ju - j + 1):
                        if abs(A_lu[kv - k, j + k]) > 1e-18:
                            for i in range(1, km + 1):
                                A_lu[kv + i - k, j + k] -= A_lu[kv + i, j] * A_lu[kv - k, j + k]

        return A_lu, pivot, info

    def solve(self, A_lu: np.ndarray, pivot: np.ndarray, b: np.ndarray, trans: str = 'N') -> np.ndarray:
        """
        求解带状 LU 分解后的线性系统 A x = b 或 A^T x = b。

        Parameters
        ----------
        A_lu : ndarray
            LU 因子。
        pivot : ndarray
            置换向量（1-based）。
        b : ndarray, shape (n,) or (n, nrhs)
            右端项。
        trans : str
            'N' 表示 A x = b，'T' 表示 A^T x = b。

        Returns
        -------
        x : ndarray
            解向量。
        """
        n = self.n
        ml = self.ml
        mu = self.mu
        kd = self.kd

        b = np.asarray(b, dtype=np.float64)
        if b.ndim == 1:
            b = b[:, np.newaxis]
            squeeze = True
        else:
            squeeze = False

        nrhs = b.shape[1]
        x = b.copy()

        if trans.upper() == 'N':
            # 解 L y = P b
            if ml > 0:
                for j in range(n - 1):
                    lm = min(ml, n - j - 1)
                    l = int(pivot[j] - 1)  # 转为 0-based
                    if l != j:
                        for i in range(nrhs):
                            t = x[l, i]
                            x[l, i] = x[j, i]
                            x[j, i] = t

                    for k in range(nrhs):
                        if x[j, k] != 0.0:
                            for i in range(1, lm + 1):
                                x[j + i, k] -= A_lu[kd + i - 1, j] * x[j, k]

            # 解 U x = y
            for k in range(nrhs):
                for j in range(n - 1, -1, -1):
                    if x[j, k] != 0.0:
                        x[j, k] /= A_lu[kd - 1, j]
                        for i in range(max(0, j - ml - mu), j):
                            x[i, k] -= A_lu[kd - 1 - j + i, j] * x[j, k]

        else:
            # 解 U^T y = b
            for k in range(nrhs):
                for j in range(n):
                    temp = x[j, k]
                    for i in range(max(0, j - ml - mu), j):
                        temp -= A_lu[kd - 1 - j + i, j] * x[i, k]
                    x[j, k] = temp / A_lu[kd - 1, j]

            # 解 L^T x = y
            if ml > 0:
                for j in range(n - 2, -1, -1):
                    lm = min(ml, n - j - 1)
                    for k in range(nrhs):
                        for i in range(1, lm + 1):
                            x[j, k] -= A_lu[kd + i - 1, j] * x[j + i, k]

                    l = int(pivot[j] - 1)
                    if l != j:
                        for k in range(nrhs):
                            t = x[l, k]
                            x[l, k] = x[j, k]
                            x[j, k] = t

        if squeeze:
            return x.ravel()
        return x


def assemble_poisson_boltzmann_jacobian(
    n: int,
    phi: np.ndarray,
    rho: np.ndarray,
    h: float,
    epsilon: float = 80.0,
    kappa: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    组装一维非线性 Poisson-Boltzmann 方程的 Jacobian 与残差。

    离散方程（有限差分）:
        -(φ_{i-1} - 2φ_i + φ_{i+1})/h^2 + κ^2 sinh(φ_i) = 4π ρ_i / ε

    Newton 迭代中的 Jacobian:
        J_{ii}   = 2/h^2 + κ^2 cosh(φ_i)
        J_{i,i±1} = -1/h^2

    本组装逻辑受 plasma_matrix 启发，处理了边界条件并构建带状结构。

    Parameters
    ----------
    n : int
        网格点数。
    phi : ndarray, shape (n,)
        当前电势猜测值 (k_B T/e)。
    rho : ndarray, shape (n,)
        电荷密度分布。
    h : float
        网格间距 (nm)。
    epsilon : float
        介电常数。
    kappa : float
        Debye-Hückel 参数 (1/nm)。

    Returns
    -------
    J : ndarray, shape (n, n)
        Jacobian 矩阵（稠密表示，可进一步转为带状）。
    F : ndarray, shape (n,)
        残差向量。
    """
    # TODO (Hole 1): 请根据一维非线性 Poisson-Boltzmann 方程的有限差分离散，
    # 组装 Jacobian 矩阵 J 和残差向量 F。
    # 离散方程: -(φ_{i-1} - 2φ_i + φ_{i+1})/h^2 + κ^2 sinh(φ_i) = 4π ρ_i / ε
    # 边界条件: Dirichlet, φ = 0 at boundaries
    # 关键科学知识点:
    #   - 有限差分 Laplacian: (φ_{i-1} - 2φ_i + φ_{i+1}) / h^2
    #   - 非线性项: κ^2 sinh(φ_i)
    #   - Jacobian 对角元: 2/h^2 + κ^2 cosh(φ_i)
    #   - Jacobian 次对角元: -1/h^2
    # 注意: 返回的 J 必须是三对角矩阵，与 BandedLU(n, ml=1, mu=1) 兼容。
    raise NotImplementedError("Hole 1: 待实现 PB Jacobian 与残差组装")


def solve_nonlinear_pb(
    n: int = 129,
    domain_length: float = 20.0,  # nm
    max_iter: int = 20,
    tol: float = 1e-8,
) -> dict:
    """
    求解一维非线性 Poisson-Boltzmann 方程。

    物理模型：DNA 双螺旋表面（负电荷）在盐溶液中的电势分布。
    电荷密度取高斯型，模拟磷酸骨架的电荷分布。

    Returns
    -------
    result : dict
        包含电势分布、迭代次数、最终残差范数。
    """
    h = domain_length / (n - 1)
    x = np.linspace(-domain_length / 2, domain_length / 2, n)

    # 电荷密度：模拟 DNA 磷酸骨架，中心高斯分布
    sigma = 1.5  # nm
    rho = -1.0 * np.exp(-x ** 2 / (2.0 * sigma ** 2))

    # 初始猜测
    phi = np.zeros(n, dtype=np.float64)

    solver = BandedLU(n, ml=1, mu=1)

    for it in range(max_iter):
        J, F = assemble_poisson_boltzmann_jacobian(n, phi, rho, h)

        # Jacobian 是三对角矩阵（ml=1, mu=1）
        A_band = solver._full_to_band(J)
        A_lu, pivot, info = solver.factorize(A_band)

        if info != 0:
            raise RuntimeError(f"LU factorization failed at iteration {it}, info={info}")

        delta_phi = solver.solve(A_lu, pivot, -F)
        phi += delta_phi

        res_norm = float(np.linalg.norm(F))
        if res_norm < tol:
            return {
                "x": x,
                "phi": phi,
                "rho": rho,
                "iterations": it + 1,
                "residual_norm": res_norm,
                "success": True,
            }

    return {
        "x": x,
        "phi": phi,
        "rho": rho,
        "iterations": max_iter,
        "residual_norm": float(np.linalg.norm(F)),
        "success": False,
    }
