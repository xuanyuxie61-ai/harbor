"""
high_order_operators.py - 高阶有限差分算子与带状矩阵模块

本模块实现阿尔芬波方程空间离散所需的高阶有限差分算子。
核心算法融合了以下种子项目的思想:
  - r8pbl (987): 对称正定带状矩阵的压缩存储和矩阵向量乘
  - clausen (187): Clenshaw 递推在高阶插值中的推广
  - companion_matrix (203): 差分算子的特征值分析

物理背景:
  阿尔芬波方程 ∂ψ/∂t = i ω_A ψ 的空间离散产生
  大稀疏线性系统。使用高阶有限差分 (4阶、6阶、8阶)
  可以在相同网格分辨率下获得更高精度，这对于准确
  捕获阿尔芬连续谱的精细结构至关重要。

核心公式:
  2N阶中心差分一阶导数:
    f'(x) ≈ (1/Δx) Σ_{k=1}^{N} c_k [f(x+kΔx) - f(x-kΔx)]

  2阶: c₁ = 1/2
  4阶: c₁ = 4/3, c₂ = -1/12
  6阶: c₁ = 3/2, c₂ = -3/20, c₃ = 1/90 (待确认)
  8阶: 8点差分系数

  二阶导数:
    f''(x) ≈ (1/Δx²) Σ c_k [f(x+kΔx) - 2f(x) + f(x-kΔx)]

  带状矩阵压缩存储 (R8PBL 格式):
    对于带宽为 ML 的 N×N 对称正定矩阵，
    仅存储 (ML+1) × N 的下三角部分。

作者: DA 博士级合成项目 PROJECT_290
"""

import numpy as np
from scipy import sparse


# ========================================================================
# 高阶有限差分系数表
# ========================================================================

def fd_coefficients_first_derivative(order):
    """
    返回 2N 阶中心差分一阶导数系数。

    对于 2N 阶精度，需要 N 个系数 c_k (k=1,...,N)，使得:
      f'(x) ≈ (1/h) Σ_{k=1}^{N} c_k [f(x+kh) - f(x-kh)] + O(h^{2N})

    系数由以下条件确定:
      Σ_{k=1}^{N} c_k k^{2j-1} = δ_{j,1} / 1,  j = 1, ..., N

    这可以写成 Vandermonde 型线性系统:
      V c = e₁
    其中 V_{j,k} = k^{2j-1}。

    参数:
      order: int, 有限差分阶数 (2, 4, 6, 8)

    返回:
      coeffs: ndarray, 差分系数数组
    """
    if order == 2:
        return np.array([0.5])
    elif order == 4:
        return np.array([2.0 / 3.0, -1.0 / 12.0])
    elif order == 6:
        return np.array([3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0])
    elif order == 8:
        return np.array([4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0])
    else:
        raise ValueError(f"不支持的有限差分阶数 {order}，仅支持 2,4,6,8")


def fd_coefficients_second_derivative(order):
    """
    返回 2N 阶中心差分二阶导数系数。

    f''(x) ≈ (1/h²) [d₀ f(x) + Σ_{k=1}^{N} d_k (f(x+kh) + f(x-kh))]

    参数:
      order: int, 有限差分阶数 (2, 4, 6, 8)

    返回:
      d0: float, 中心系数
      coeffs: ndarray, 非中心系数
    """
    if order == 2:
        return -2.0, np.array([1.0])
    elif order == 4:
        return -5.0 / 2.0, np.array([4.0 / 3.0, -1.0 / 12.0])
    elif order == 6:
        return -49.0 / 18.0, np.array([3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0])
    elif order == 8:
        return -205.0 / 72.0, np.array([8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0])
    else:
        raise ValueError(f"不支持的有限差分阶数 {order}")


# ========================================================================
# 带状矩阵存储 (改编自 r8pbl 项目)
# ========================================================================

class BandMatrixSPD:
    """
    对称正定带状矩阵的压缩存储与运算。

    改编自 r8pbl (987) 项目的 R8PBL 格式:
      对于 N×N 对称矩阵，带宽 ML (半带宽)，
      仅存储 (ML+1) × N 的下三角带状部分。

    存储布局:
      A_data[0, j] = A(j, j)          对角元
      A_data[k, j] = A(j+k, j)        第 k 条次对角元

    在阿尔芬波隐式时间推进中，需求解:
      (I - Δt L) ψ^{n+1} = ψ^n
    其中 L 为空间离散算子，产生带状矩阵。

    参数:
      n: int, 矩阵阶数
      ml: int, 半带宽
    """

    def __init__(self, n, ml):
        if n <= 0:
            raise ValueError(f"矩阵阶数 n = {n} 必须为正")
        if ml < 0 or ml >= n:
            raise ValueError(f"半带宽 ml = {ml} 不合法")
        self.n = n
        self.ml = ml
        self.data = np.zeros((ml + 1, n), dtype=np.float64)

    def set_diagonal(self, values):
        """设置对角元 A(j,j) = values[j]。"""
        values = np.asarray(values, dtype=np.float64)
        if len(values) != self.n:
            raise ValueError(f"对角元长度 {len(values)} ≠ n = {self.n}")
        self.data[0, :] = values

    def set_subdiagonal(self, k, values):
        """设置第 k 条次对角元 A(j+k, j) = values[j]。"""
        if k < 1 or k > self.ml:
            raise ValueError(f"次对角编号 k = {k} 超出带宽 ml = {self.ml}")
        values = np.asarray(values, dtype=np.float64)
        expected_len = self.n - k
        if len(values) != expected_len:
            raise ValueError(f"第 {k} 条次对角元长度 {len(values)} ≠ {expected_len}")
        self.data[k, :self.n - k] = values

    def mv(self, x):
        """
        对称带状矩阵向量乘法 b = A x。

        利用对称性: A(i,j) = A(j,i)，
        对于第 k 条次对角元:
          b(j) += A(j+k, j) * x(j)    (下三角)
          b(j+k) += A(j+k, j) * x(j)  (上三角，利用对称)

        改编自 r8pbl_mv。

        参数:
          x: ndarray, 输入向量

        返回:
          b: ndarray, 结果向量
        """
        x = np.asarray(x, dtype=np.float64)
        if len(x) != self.n:
            raise ValueError(f"输入向量长度 {len(x)} ≠ n = {self.n}")

        b = np.zeros(self.n, dtype=np.float64)

        # 对角元贡献
        b += self.data[0, :] * x

        # 次对角元贡献 (利用对称性)
        for k in range(1, self.ml + 1):
            n_k = self.n - k
            aij = self.data[k, :n_k]
            # 下三角: b(j) += A(j+k, j) * x(j)
            b[:n_k] += aij * x[:n_k]
            # 上三角: b(j+k) += A(j+k, j) * x(j+k) = A(j, j+k) * x(j+k)
            b[k:k + n_k] += aij * x[k:k + n_k]

        return b

    def to_dense(self):
        """
        将压缩带状格式转换为全密度矩阵。

        改编自 r8pbl_to_r8ge。

        返回:
          A: ndarray, shape (n, n), 全密度矩阵
        """
        A = np.zeros((self.n, self.n), dtype=np.float64)

        for j in range(self.n):
            A[j, j] = self.data[0, j]
            for k in range(1, self.ml + 1):
                if j + k < self.n:
                    val = self.data[k, j]
                    A[j + k, j] = val
                    A[j, j + k] = val

        return A

    def to_scipy_sparse(self):
        """
        转换为 scipy.sparse 格式用于高效求解。
        """
        A_dense = self.to_dense()
        return sparse.csr_matrix(A_dense)

    @staticmethod
    def create_dif2(n, ml=1):
        """
        创建二阶差分矩阵 (三对角 [-1, 2, -1])。

        这是阿尔芬波方程空间离散的最基本算子。
        对应 ∂²ψ/∂r² 的二阶中心差分近似。

        改编自 r8pbl_dif2。

        参数:
          n: int, 矩阵阶数
          ml: int, 半带宽 (默认 1)

        返回:
          BandMatrixSPD, 二阶差分矩阵
        """
        if ml < 1:
            ml = 1
        A = BandMatrixSPD(n, ml)
        A.set_diagonal(2.0 * np.ones(n))
        if ml >= 1:
            A.set_subdiagonal(1, -np.ones(n - 1))
        return A

    @staticmethod
    def create_high_order_laplacian(n, order, ml=None):
        """
        创建高阶拉普拉斯算子矩阵。

        对于 2N 阶差分近似的二阶导数:
          L_{j,j} = d₀
          L_{j,j±k} = d_k,  k = 1, ..., N

        在阿尔芬波方程中，这个算子用于:
          ∂²ψ/∂r² ≈ (1/Δr²) L ψ

        参数:
          n: int, 矩阵阶数
          order: int, 有限差分阶数
          ml: int or None, 半带宽 (默认自动)

        返回:
          BandMatrixSPD, 高阶拉普拉斯矩阵
        """
        d0, coeffs = fd_coefficients_second_derivative(order)
        N_coeffs = len(coeffs)

        if ml is None:
            ml = N_coeffs

        A = BandMatrixSPD(n, ml)
        A.set_diagonal(d0 * np.ones(n))

        for k in range(N_coeffs):
            if k + 1 <= ml:
                A.set_subdiagonal(k + 1, coeffs[k] * np.ones(n - k - 1))

        return A


# ========================================================================
# 有限差分算子应用
# ========================================================================

def apply_first_derivative(field, dx, order, periodic=True):
    """
    对一维场施加高阶一阶导数有限差分。

    f'(x_i) ≈ (1/Δx) Σ_{k=1}^{N} c_k [f(x_{i+k}) - f(x_{i-k})]

    参数:
      field: ndarray, 一维场数据
      dx: float, 网格间距
      order: int, 差分阶数
      periodic: bool, 是否使用周期边界

    返回:
      deriv: ndarray, 导数近似
    """
    field = np.asarray(field, dtype=np.float64)
    n = len(field)
    coeffs = fd_coefficients_first_derivative(order)
    N_stencil = len(coeffs)

    deriv = np.zeros(n, dtype=np.float64)

    for i in range(n):
        for k_idx in range(N_stencil):
            k = k_idx + 1
            if periodic:
                i_plus = (i + k) % n
                i_minus = (i - k) % n
            else:
                i_plus = min(i + k, n - 1)
                i_minus = max(i - k, 0)
            deriv[i] += coeffs[k_idx] * (field[i_plus] - field[i_minus])

    deriv /= dx
    return deriv


def apply_second_derivative(field, dx, order, periodic=True):
    """
    对一维场施加高阶二阶导数有限差分。

    f''(x_i) ≈ (1/Δx²) [d₀ f(x_i) + Σ c_k (f(x_{i+k}) + f(x_{i-k}))]

    参数:
      field: ndarray, 一维场数据
      dx: float, 网格间距
      order: int, 差分阶数
      periodic: bool, 是否使用周期边界

    返回:
      deriv2: ndarray, 二阶导数近似
    """
    field = np.asarray(field, dtype=np.float64)
    n = len(field)
    d0, coeffs = fd_coefficients_second_derivative(order)
    N_stencil = len(coeffs)

    deriv2 = np.zeros(n, dtype=np.float64)

    for i in range(n):
        deriv2[i] = d0 * field[i]
        for k_idx in range(N_stencil):
            k = k_idx + 1
            if periodic:
                i_plus = (i + k) % n
                i_minus = (i - k) % n
            else:
                i_plus = min(i + k, n - 1)
                i_minus = max(i - k, 0)
            deriv2[i] += coeffs[k_idx] * (field[i_plus] + field[i_minus])

    deriv2 /= dx ** 2
    return deriv2


def apply_radial_laplacian_cylindrical(field, r_array, dr, order):
    """
    柱坐标下的径向拉普拉斯算子:

    ∇²_r f = (1/r) ∂/∂r (r ∂f/∂r) = ∂²f/∂r² + (1/r) ∂f/∂r

    在环形等离子体中，对于 n=0 的轴对称模式，
    径向拉普拉斯采用柱坐标形式。

    参数:
      field: ndarray, 径向场数据
      r_array: ndarray, 径向坐标数组
      dr: float, 径向间距
      order: int, 差分阶数

    返回:
      laplacian: ndarray, 柱坐标径向拉普拉斯
    """
    df_dr = apply_first_derivative(field, dr, order, periodic=False)
    d2f_dr2 = apply_second_derivative(field, dr, order, periodic=False)

    # 避免 r=0 处的奇点
    r_safe = np.where(r_array > 1e-14, r_array, 1e-14)
    laplacian = d2f_dr2 + df_dr / r_safe

    # 在 r=0 处使用 L'Hôpital 规则: ∇²_r f(r=0) = 2 f''(0)
    if r_array[0] < 1e-14:
        laplacian[0] = 2.0 * d2f_dr2[0]

    return laplacian


def truncation_error_analysis(dx, order, func, dfunc_exact, x_array):
    """
    分析有限差分近似的截断误差。

    对于 2N 阶差分，截断误差为:
      E = C_{2N} h^{2N} f^{(2N+1)}(ξ)

    本函数通过比较数值导数和精确导数来验证收敛阶。

    参数:
      dx: float, 网格间距
      order: int, 差分阶数
      func: callable, 被微分函数
      dfunc_exact: callable, 精确导数函数
      x_array: ndarray, 评估点

    返回:
      error_l2: float, L2 范数误差
      error_linf: float, L∞ 范数误差
      convergence_order: float, 估计的收敛阶
    """
    field = func(x_array)
    deriv_num = apply_first_derivative(field, dx, order, periodic=False)
    deriv_exact = dfunc_exact(x_array)

    error = deriv_num - deriv_exact
    error_l2 = np.sqrt(np.mean(error ** 2))
    error_linf = np.max(np.abs(error))

    return error_l2, error_linf
