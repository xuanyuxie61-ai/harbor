"""
utils.py — 数值工具模块
========================
提供项目共用的数值方法: 二分法搜索最优阶数、
稀疏矩阵装配、FFT 微分算子等。

映射种子项目:
  - 096_bisection_min: 二分法搜索最优 PCE 阶数
  - 986_r8ncf: 稀疏矩阵 COO 格式
  - 738_matrix_assemble_parfor: 并行矩阵装配
"""

import numpy as np
from scipy import sparse
from typing import Callable, Tuple, Optional
import itertools


# ============================================================
#  二分法搜索最优多项式阶数 (映射 096_bisection_min)
# ============================================================
def bisection_optimal_order(error_func: Callable[[int], float],
                            p_min: int,
                            p_max: int,
                            target_error: float,
                            max_iter: int = 50) -> int:
    """
    使用二分法搜索满足目标误差的最小多项式阶数。

    策略: 在 [p_min, p_max] 上搜索, 找到使得
      error_func(p) <= target_error
    成立的最小 p。

    映射 096_bisection_min: 将单峰函数最小值搜索
    改为单调递减误差函数的根查找。

    参数:
        error_func:    输入阶数 p, 返回误差估计
        p_min:         搜索下界
        p_max:         搜索上界
        target_error:  目标误差容限
        max_iter:      最大迭代次数

    返回:
        p_opt: 满足条件的最小阶数
    """
    lo, hi = p_min, p_max

    # 检查边界
    if error_func(lo) <= target_error:
        return lo
    if error_func(hi) > target_error:
        return hi  # 即使最高阶也不满足

    for iteration in range(max_iter):
        if hi - lo <= 1:
            return hi

        mid = (lo + hi) // 2
        err = error_func(mid)

        if err <= target_error:
            hi = mid  # 可以尝试更低的阶
        else:
            lo = mid  # 需要更高的阶

    return hi


# ============================================================
#  稀疏矩阵 COO 格式装配 (映射 986_r8ncf)
# ============================================================
class SparseMatrixCOO:
    """
    COO 格式稀疏矩阵, 用于随机 Galerkin 系统装配。

    映射 986_r8ncf: 使用 (row, col, val) 三元组存储
    稀疏矩阵的非零元素, 支持增量装配和转换为 CSR。

    属性:
        rows: 行索引数组
        cols: 列索引数组
        vals: 值数组
        shape: 矩阵形状 (m, n)
    """

    def __init__(self, m: int, n: int):
        self.rows: list = []
        self.cols: list = []
        self.vals: list = []
        self.shape = (m, n)
        self._nnz = 0

    def add(self, row: int, col: int, val: float):
        """添加一个非零元素"""
        if abs(val) > 1e-16:
            self.rows.append(row)
            self.cols.append(col)
            self.vals.append(val)
            self._nnz += 1

    def add_block(self, rows: np.ndarray, cols: np.ndarray,
                  vals: np.ndarray):
        """批量添加非零元素"""
        mask = np.abs(vals) > 1e-16
        self.rows.extend(rows[mask].tolist())
        self.cols.extend(cols[mask].tolist())
        self.vals.extend(vals[mask].tolist())
        self._nnz += int(np.sum(mask))

    def to_scipy_sparse(self) -> sparse.csr_matrix:
        """转换为 scipy CSR 稀疏矩阵"""
        if self._nnz == 0:
            return sparse.csr_matrix(self.shape)
        return sparse.coo_matrix(
            (np.array(self.vals),
             (np.array(self.rows, dtype=int),
              np.array(self.cols, dtype=int))),
            shape=self.shape
        ).tocsr()

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵 (仅用于调试/小规模)"""
        return self.to_scipy_sparse().toarray()

    @property
    def nnz(self) -> int:
        return self._nnz


# ============================================================
#  并行矩阵装配 (映射 738_matrix_assemble_parfor)
# ============================================================
def assemble_stochastic_galerkin_matrix(
    coupling_tensor: np.ndarray,
    diffusion_coeffs: np.ndarray,
    n_physical_dof: int
) -> sparse.csr_matrix:
    """
    并行装配随机 Galerkin 系统矩阵。

    映射 738_matrix_assemble_parfor: 将 Hilbert 矩阵的并行装配
    推广到随机 Galerkin 矩阵的块结构装配。

    系统矩阵结构:
      A_{ij}^{SG} = Σ_k c_k * C_{ijk} * K^{phys}

    其中:
      c_k: 扩散系数 (PCE 系数)
      C_{ijk}: 三重乘积张量 <Ψ_i Ψ_j, Ψ_k>
      K^{phys}: 物理空间刚度矩阵

    参数:
        coupling_tensor: 三重乘积张量, shape (P, P, P)
        diffusion_coeffs: 扩散系数的 PCE 系数, shape (P,)
        n_physical_dof: 物理空间自由度数

    返回:
        A: 系统矩阵 (P*n_physical_dof, P*n_physical_dof), CSR 格式
    """
    P = coupling_tensor.shape[0]
    N_total = P * n_physical_dof

    mat = SparseMatrixCOO(N_total, N_total)

    # 对每个 PCE 阶 k, 装配贡献
    for k in range(P):
        if abs(diffusion_coeffs[k]) < 1e-16:
            continue

        # 块矩阵 (i, j) 的系数 = Σ_k c_k * C_{ijk}
        for i in range(P):
            for j in range(P):
                coeff = diffusion_coeffs[k] * coupling_tensor[i, j, k]
                if abs(coeff) < 1e-16:
                    continue

                # 块 (i, j): 对应物理矩阵的 (n_phys × n_phys) 子块
                # 简化: 使用单位矩阵作为物理空间耦合
                for dof in range(n_physical_dof):
                    row_global = i * n_physical_dof + dof
                    col_global = j * n_physical_dof + dof
                    mat.add(row_global, col_global, coeff)

    return mat.to_scipy_sparse()


# ============================================================
#  二维谱微分算子 (FFT 方法)
# ============================================================
def spectral_derivative_2d(field: np.ndarray,
                           Lx: float, Ly: float
                           ) -> Tuple[np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
    """
    使用 FFT 计算二维周期域上的谱导数。

    对 f(x, y) 在 [0, Lx] × [0, Ly] 上:
      ∂f/∂x = IFFT(i kx * FFT(f))
      ∂f/∂y = IFFT(i ky * FFT(f))
      ∇²f = IFFT(-(kx² + ky²) * FFT(f))

    参数:
        field: shape (ny, nx) 的场
        Lx, Ly: 域尺寸

    返回:
        dfdx: ∂f/∂x
        dfdy: ∂f/∂y
        laplacian: ∇²f
        biharmonic: ∇⁴f
    """
    ny, nx = field.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    KX, KY = np.meshgrid(kx, ky)

    K2 = KX ** 2 + KY ** 2     # |k|^2
    K4 = K2 ** 2                # |k|^4

    f_hat = np.fft.fft2(field)

    dfdx = np.real(np.fft.ifft2(1j * KX * f_hat))
    dfdy = np.real(np.fft.ifft2(1j * KY * f_hat))
    laplacian = np.real(np.fft.ifft2(-K2 * f_hat))
    biharmonic = np.real(np.fft.ifft2(K4 * f_hat))

    return dfdx, dfdy, laplacian, biharmonic


# ============================================================
#  多维索引枚举
# ============================================================
def multi_index_set(d: int, p: int,
                    truncation: str = "total_order",
                    q: float = 1.0
                    ) -> np.ndarray:
    """
    生成多维多项式索引集。

    total_order:  {i ∈ N^d : |i|_1 = i_1+...+i_d <= p}
    hyperbolic:   {i ∈ N^d : (Σ i_k^q)^{1/q} <= p}

    参数:
        d: 维度
        p: 最大阶数
        truncation: 截断类型
        q: 双曲截断参数 (仅 hyperbolic)

    返回:
        indices: shape (P, d) 的整数数组, P = |索引集|
    """
    if truncation == "total_order":
        indices = []
        for multi_idx in itertools.product(range(p + 1), repeat=d):
            if sum(multi_idx) <= p:
                indices.append(multi_idx)
        return np.array(indices, dtype=int)

    elif truncation == "hyperbolic":
        indices = []
        for multi_idx in itertools.product(range(p + 1), repeat=d):
            # 防止 0^q 的问题
            norm_q = sum(max(k, 0) ** q for k in multi_idx)
            if norm_q ** (1.0 / q) <= p + 1e-10:
                indices.append(multi_idx)
        return np.array(indices, dtype=int)

    raise ValueError(f"Unknown truncation: {truncation}")


# ============================================================
#  条件数估计与数值稳定性检查
# ============================================================
def check_condition_number(A: np.ndarray,
                           name: str = "matrix") -> float:
    """
    估计矩阵条件数并检查数值稳定性。

    参数:
        A: 方阵
        name: 矩阵名称 (用于日志)

    返回:
        cond: 条件数
    """
    cond = np.linalg.cond(A)
    if cond > 1e12:
        print(f"  [WARNING] {name} 条件数过大: {cond:.2e}")
    elif cond > 1e8:
        print(f"  [INFO] {name} 条件数: {cond:.2e}")
    return cond


# ============================================================
#  数值安全除法
# ============================================================
def safe_divide(a: np.ndarray, b: np.ndarray,
                eps: float = 1e-15) -> np.ndarray:
    """
    安全除法: 防止除以零。

    a / max(|b|, eps) * sign(b)
    """
    sign_b = np.sign(b)
    sign_b[sign_b == 0] = 1.0
    return a / np.maximum(np.abs(b), eps) * sign_b


# ============================================================
#  一维网格生成 (映射 680_line_grid)
# ============================================================
def line_grid(n_points: int, a: float, b: float,
              centering: int = 1) -> np.ndarray:
    """
    生成一维线段 [a, b] 上的网格点。

    映射 680_line_grid: 支持多种居中方式。

    centering:
      1: 包含两端点
      2: 不包含端点
      3: 仅包含左端点
      4: 仅包含右端点
      5: 半整数中点

    参数:
        n_points:  网格点数
        a, b:      区间端点
        centering: 居中方式

    返回:
        x: shape (n_points,), 网格坐标
    """
    if n_points < 1:
        return np.array([])

    if n_points == 1:
        return np.array([0.5 * (a + b)])

    if centering == 1:
        return np.linspace(a, b, n_points)
    elif centering == 2:
        h = (b - a) / (n_points + 1)
        return a + h + h * np.arange(n_points)
    elif centering == 3:
        h = (b - a) / n_points
        return a + h * np.arange(n_points)
    elif centering == 4:
        h = (b - a) / n_points
        return a + h * np.arange(1, n_points + 1)
    elif centering == 5:
        h = (b - a) / n_points
        return a + 0.5 * h + h * np.arange(n_points)
    else:
        raise ValueError(f"Invalid centering: {centering}")
