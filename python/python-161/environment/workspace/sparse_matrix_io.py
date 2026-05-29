"""
sparse_matrix_io.py
基于种子项目 769_mm_io (Matrix Market file I/O)
改造为钙钛矿太阳能电池漂移-扩散方程离散后的大型稀疏矩阵存储/读取模块。

在有限差分/有限元离散漂移-扩散方程后，得到的 Jacobian 矩阵通常为
大型稀疏矩阵（维度 10^4–10^6，非零元密度 < 0.1%）。
Matrix Market (MM) 格式是科研计算中广泛使用的稀疏矩阵交换格式。

核心公式：
  1. 一维有限差分离散（Scharfetter-Gummel 格式）：
       J_n = q * μ_n * n * E + q * D_n * dn/dx
     其中 D_n = μ_n * k_B * T / q （Einstein 关系）。
  2. 离散后的线性系统：A * φ = b
     A 为 N×N 稀疏矩阵，采用 coordinate (COO) 格式存储：
       MM 头：%%MatrixMarket matrix coordinate real general
       每行：row_index col_index value
  3. 对称矩阵可标记为 'symmetric' 以节省存储。
"""

import numpy as np
from typing import Tuple, List


class SparseMatrix:
    """
    基于 COO 格式的稀疏矩阵，支持 Matrix Market 格式读写。
    """

    def __init__(self, nrow: int, ncol: int):
        self.nrow = nrow
        self.ncol = ncol
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[float] = []

    def add(self, i: int, j: int, v: float) -> None:
        """添加非零元（允许重复，后续可累加）。"""
        if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
            raise IndexError(f"索引 ({i},{j}) 超出矩阵维度 ({self.nrow},{self.ncol})")
        self.rows.append(i)
        self.cols.append(j)
        self.vals.append(v)

    def to_dense(self) -> np.ndarray:
        """转为稠密矩阵（仅用于小规模调试）。"""
        A = np.zeros((self.nrow, self.ncol))
        for i, j, v in zip(self.rows, self.cols, self.vals):
            A[i, j] += v
        return A

    def to_coo_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 COO 三元组。"""
        return np.array(self.rows), np.array(self.cols), np.array(self.vals)

    def nnz(self) -> int:
        return len(self.rows)


def build_drift_diffusion_jacobian(
    N: int,
    dx: float,
    mu_n: float,
    mu_p: float,
    D_n: float,
    D_p: float,
    E_field: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    kT_q: float,
) -> SparseMatrix:
    """
    构建一维漂移-扩散方程稳态 Newton 迭代 Jacobian 的稀疏矩阵。

    方程组（电子连续性、空穴连续性、Poisson）：
      dJ_n/dx = q * (G - R)
      dJ_p/dx = -q * (G - R)
      d²φ/dx² = -q/ε * (p - n + N_D^+ - N_A^-)

    这里简化构建耦合系统的线性化矩阵块结构。

    Parameters
    ----------
    N : int
        空间格点数
    dx : float
        网格间距 [cm]
    mu_n, mu_p : float
        电子/空穴迁移率 [cm^2/(V·s)]
    D_n, D_p : float
        电子/空穴扩散系数 [cm^2/s]
    E_field : (N,) array
        电场 [V/cm]
    n, p : (N,) array
        电子/空穴浓度 [cm^{-3}]
    kT_q : float
        热电压 k_B*T/q [V]

    Returns
    -------
    jac : SparseMatrix
        (3N) × (3N) 稀疏 Jacobian
    """
    if N <= 2:
        raise ValueError("格点数必须大于 2")
    if dx <= 0:
        raise ValueError("网格间距必须为正")

    jac = SparseMatrix(3 * N, 3 * N)

    # 电子连续性方程（指标 0:N 对应 n）
    for i in range(1, N - 1):
        # Scharfetter-Gummel 离散：
        # J_n,i+1/2 = (q*D_n/dx) * [B(Δφ) * n_i - B(-Δφ) * n_{i+1}]
        # 其中 B(x) = x / (exp(x)-1) 为 Bernoulli 函数
        dphi = E_field[i] * dx / kT_q if kT_q != 0 else 0.0
        B_plus = _bernoulli(dphi)
        B_minus = _bernoulli(-dphi)

        coeff = D_n / (dx * dx)
        # ∂F_ni/∂n_i
        jac.add(N + i, i, coeff * B_plus)
        # ∂F_ni/∂n_{i-1}
        jac.add(N + i, i - 1, -coeff * B_minus)
        # ∂F_ni/∂n_{i+1}
        jac.add(N + i, i + 1, -coeff * B_plus)
        # 对角耦合到 Poisson（简化）
        jac.add(N + i, 2 * N + i, 1.0e-10)

    # 空穴连续性方程（指标 N:2N 对应 p）
    for i in range(1, N - 1):
        dphi = E_field[i] * dx / kT_q if kT_q != 0 else 0.0
        B_plus = _bernoulli(dphi)
        B_minus = _bernoulli(-dphi)

        coeff = D_p / (dx * dx)
        # ∂F_pi/∂p_i
        jac.add(2 * N + i, N + i, coeff * B_plus)
        # ∂F_pi/∂p_{i-1}
        jac.add(2 * N + i, N + i - 1, -coeff * B_minus)
        # ∂F_pi/∂p_{i+1}
        jac.add(2 * N + i, N + i + 1, -coeff * B_plus)
        jac.add(2 * N + i, 2 * N + i, 1.0e-10)

    # Poisson 方程（指标 2N:3N 对应 φ）
    eps = 1.0e-12  # 介电常数缩放 [F/cm]
    for i in range(1, N - 1):
        # 二阶差分 (φ_{i-1} - 2φ_i + φ_{i+1}) / dx^2
        jac.add(i, 2 * N + i - 1, 1.0 / (dx * dx))
        jac.add(i, 2 * N + i, -2.0 / (dx * dx))
        jac.add(i, 2 * N + i + 1, 1.0 / (dx * dx))
        # 耦合到载流子浓度
        jac.add(i, i, -1.0 / eps)
        jac.add(i, N + i, 1.0 / eps)

    # Dirichlet 边界条件
    for i in [0, N - 1]:
        jac.add(N + i, i, 1.0)   # n 边界
        jac.add(2 * N + i, N + i, 1.0)  # p 边界
        jac.add(i, 2 * N + i, 1.0)  # φ 边界

    return jac


def _bernoulli(x: float) -> float:
    """
    Bernoulli 函数 B(x) = x / (exp(x) - 1)。
    数值稳定实现：
      x → 0 时，B(x) ≈ 1 - x/2 + x^2/12
      x → -∞ 时，B(x) ≈ -x
      x → +∞ 时，B(x) ≈ 0
    """
    if abs(x) < 1e-5:
        return 1.0 - x / 2.0 + x * x / 12.0
    elif x > 20.0:
        return x * np.exp(-x)
    elif x < -20.0:
        return -x
    else:
        return x / (np.exp(x) - 1.0)


def write_matrix_market(A: SparseMatrix, filename: str, symm: str = "general") -> None:
    """
    将稀疏矩阵写入 Matrix Market 格式文件。
    """
    with open(filename, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real {}\n".format(symm))
        f.write("% Generated by sparse_matrix_io.py for perovskite solar cell\n")
        f.write("{} {} {}\n".format(A.nrow, A.ncol, A.nnz()))
        for i, j, v in zip(A.rows, A.cols, A.vals):
            f.write("{} {} {:.16e}\n".format(i + 1, j + 1, v))


def read_matrix_market(filename: str) -> SparseMatrix:
    """
    从 Matrix Market 格式文件读取稀疏矩阵。
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 跳过注释行
    idx = 0
    while lines[idx].strip().startswith('%'):
        idx += 1

    header = lines[idx].strip().split()
    nrow, ncol, nnz_expected = int(header[0]), int(header[1]), int(header[2])
    A = SparseMatrix(nrow, ncol)

    for line in lines[idx + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        i, j, v = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
        A.add(i, j, v)

    if A.nnz() != nnz_expected:
        # 允许对称矩阵只存一半的情况
        pass
    return A


if __name__ == "__main__":
    N = 10
    dx = 1e-6
    E = np.zeros(N)
    n = np.ones(N) * 1e15
    p = np.ones(N) * 1e15
    jac = build_drift_diffusion_jacobian(N, dx, 20.0, 10.0, 0.5, 0.25, E, n, p, 0.0259)
    print(f"Jacobian 维度: {jac.nrow}×{jac.ncol}, 非零元: {jac.nnz()}")
    fname = "/mnt/data/zpy/sci-swe/source code/Synthesis-project-python/161_synth_project/test_jac.mtx"
    write_matrix_market(jac, fname)
    A2 = read_matrix_market(fname)
    print(f"读取后非零元: {A2.nnz()}")
    import os
    os.remove(fname)
