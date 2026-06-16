"""
高阶有限差分算子模块
实现 Weyl 半金属 Hamiltonian 在 k 空间的高精度差分格式

核心数学：
有限差分法将微分算子离散化：
∂f/∂x ≈ Σ_i c_i · f(x + i·h)

精度分析：
O(h²): [-f(x+h) + f(x-h)] / (2h)
O(h⁴): [f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)] / (12h)
O(h⁶): [-f(x-3h) + 9f(x-2h) - 45f(x-h) + 45f(x+h) - 9f(x+2h) + f(x+3h)] / (60h)
"""

import numpy as np
from typing import Tuple, List, Callable


class HighOrderFDScheme:
    """
    高阶有限差分格式管理器

    针对 Weyl Hamiltonian 的 k 空间微分，
    提供不同精度和带宽的差分格式
    """

    # 预计算的差分系数 (中心差分)
    FD_COEFFICIENTS = {
        2: {  # O(h²)
            'stencil': [-1, 0, 1],
            'weights': np.array([-0.5, 0.0, 0.5]),
            'order': 2
        },
        4: {  # O(h⁴)
            'stencil': [-2, -1, 0, 1, 2],
            'weights': np.array([1/12, -8/12, 0.0, 8/12, -1/12]),
            'order': 4
        },
        6: {  # O(h⁶)
            'stencil': [-3, -2, -1, 0, 1, 2, 3],
            'weights': np.array([-1/60, 9/60, -45/60, 0.0, 45/60, -9/60, 1/60]),
            'order': 6
        },
        8: {  # O(h⁸)
            'stencil': [-4, -3, -2, -1, 0, 1, 2, 3, 4],
            'weights': np.array([1/280, -4/105, 1/5, -4/5, 0.0, 4/5, -1/5, 4/105, -1/280]),
            'order': 8
        }
    }

    def __init__(self, order: int = 4):
        """
        初始化差分格式

        Parameters:
        -----------
        order : int
            差分精度阶数
        """
        if order not in self.FD_COEFFICIENTS:
            raise ValueError(f"Order {order} not supported. "
                           f"Available: {list(self.FD_COEFFICIENTS.keys())}")
        self.order = order
        self.scheme = self.FD_COEFFICIENTS[order]

    def derivative_1d(self, f_values: np.ndarray, dx: float) -> np.ndarray:
        """
        一维函数的一阶导数

        Parameters:
        -----------
        f_values : np.ndarray
            函数值数组
        dx : float
            步长

        Returns:
        --------
        np.ndarray
            导数值数组
        """
        weights = self.scheme['weights']
        stencil = self.scheme['stencil']
        n = len(f_values)
        df = np.zeros(n)

        for i in range(n):
            val = 0.0
            for w, s in zip(weights, stencil):
                j = i + s
                # 周期性边界条件 (Brillouin 区)
                j = j % n
                val += w * f_values[j]
            df[i] = val / dx

        return df

    def derivative_3d(self, f_func: Callable, k_point: np.ndarray,
                      direction: int, dk: float) -> float:
        """
        三维函数沿某方向的方向导数

        Parameters:
        -----------
        f_func : Callable
            函数 f(kx, ky, kz) → scalar or matrix
        k_point : np.ndarray
            当前 k 点
        direction : int
            方向 (0=x, 1=y, 2=z)
        dk : float
            步长

        Returns:
        --------
        float or np.ndarray
            方向导数值
        """
        weights = self.scheme['weights']
        stencil = self.scheme['stencil']

        result = None
        for w, s in zip(weights, stencil):
            k_shifted = k_point.copy()
            k_shifted[direction] += s * dk
            val = f_func(k_shifted)

            if result is None:
                result = w * val
            else:
                result = result + w * val

        return result / dk

    def laplacian_3d(self, f_func: Callable, k_point: np.ndarray,
                     dk: float) -> float:
        """
        三维 Laplacian (∂²/∂kx² + ∂²/∂ky² + ∂²/∂kz²)

        使用 O(h²) 七点格式：
        ∇²f ≈ [f(k+h·x) + f(k-h·x) + f(k+h·y) + f(k-h·y) +
               f(k+h·z) + f(k-h·z) - 6f(k)] / h²

        Parameters:
        -----------
        f_func : Callable
            标量函数
        k_point : np.ndarray
            当前 k 点
        dk : float
            步长

        Returns:
        --------
        float
            Laplacian 值
        """
        f_center = f_func(k_point)
        laplacian = -6.0 * f_center

        for d in range(3):
            k_plus = k_point.copy()
            k_plus[d] += dk
            k_minus = k_point.copy()
            k_minus[d] -= dk

            laplacian += f_func(k_plus) + f_func(k_minus)

        return laplacian / (dk**2)

    def curl_fd(self, vector_field: Callable, k_point: np.ndarray,
                dk: float) -> np.ndarray:
        """
        矢量场的旋度 (∇ × A)

        (∇ × A)_x = ∂Az/∂ky - ∂Ay/∂kz
        ...

        Parameters:
        -----------
        vector_field : Callable
            矢量场函数 k → (Ax, Ay, Az)
        k_point : np.ndarray
            当前 k 点
        dk : float
            步长

        Returns:
        --------
        np.ndarray
            旋度 (curl_x, curl_y, curl_z)
        """
        # 各方向偏导数
        dAx_dy = self.derivative_3d(lambda k: vector_field(k)[0], k_point, 1, dk)
        dAx_dz = self.derivative_3d(lambda k: vector_field(k)[0], k_point, 2, dk)
        dAy_dx = self.derivative_3d(lambda k: vector_field(k)[1], k_point, 0, dk)
        dAy_dz = self.derivative_3d(lambda k: vector_field(k)[1], k_point, 2, dk)
        dAz_dx = self.derivative_3d(lambda k: vector_field(k)[2], k_point, 0, dk)
        dAz_dy = self.derivative_3d(lambda k: vector_field(k)[2], k_point, 1, dk)

        curl_x = dAz_dy - dAy_dz
        curl_y = dAx_dz - dAz_dx
        curl_z = dAy_dx - dAx_dy

        return np.array([curl_x, curl_y, curl_z])

    def truncation_error_estimate(self, f_func: Callable, k_point: np.ndarray,
                                   direction: int, dk: float,
                                   dk_refine: float = None) -> float:
        """
        估计截断误差

        通过网格细化比较来估计实际误差：
        error ≈ |D_h^{(p)} f - D_{h/2}^{(p)} f| / (2^p - 1)

        Parameters:
        -----------
        f_func : Callable
            函数
        k_point : np.ndarray
            k 点
        direction : int
            方向
        dk : float
            原始步长
        dk_refine : float
            细化步长 (默认 dk/2)

        Returns:
        --------
        float
            估计的截断误差
        """
        if dk_refine is None:
            dk_refine = dk / 2

        df_coarse = self.derivative_3d(f_func, k_point, direction, dk)
        df_fine = self.derivative_3d(f_func, k_point, direction, dk_refine)

        # Richardson 外推误差估计
        p = self.order
        error = np.abs(df_fine - df_coarse) / (2**p - 1)
        if isinstance(error, np.ndarray):
            error = np.linalg.norm(error)

        return error


class FDMatrixOperator:
    """
    有限差分矩阵算子

    将微分算子表示为矩阵形式，用于 Hamiltonian 的矩阵表示
    """

    def __init__(self, n_grid: int, period: float = 2*np.pi):
        """
        初始化

        Parameters:
        -----------
        n_grid : int
            网格点数
        period : float
            周期
        """
        self.n_grid = n_grid
        self.period = period
        self.dk = period / n_grid

    def momentum_operator(self, direction: int = 0) -> np.ndarray:
        """
        构建动量算符 k_i 的矩阵表示

        在平面波基底下：k_i → -i∂/∂x_i

        Parameters:
        -----------
        direction : int
            方向

        Returns:
        --------
        np.ndarray
            动量算符矩阵 (n×n)
        """
        n = self.n_grid
        k_operator = np.zeros((n, n), dtype=complex)

        # 使用四阶差分
        stencil = [-2, -1, 1, 2]
        weights = np.array([1/12, -8/12, 8/12, -1/12]) / self.dk

        for i in range(n):
            for w, s in zip(weights, stencil):
                j = (i + s) % n
                k_operator[i, j] += -1j * w  # -i ∂/∂k

        return k_operator

    def kinetic_energy_matrix(self) -> np.ndarray:
        """
        构建动能算符 ℏ²k²/(2m) 的矩阵

        Returns:
        --------
        np.ndarray
            动能矩阵
        """
        k_op = self.momentum_operator(0)
        k_squared = k_op @ k_op
        return k_squared

    def weyl_hamiltonian_matrix(self, v_f: float = 1.0,
                                 m0: float = 0.1) -> np.ndarray:
        """
        构建 Weyl Hamiltonian 的有限差分矩阵

        2×2 块矩阵结构，每个块为 n×n 差分矩阵

        Parameters:
        -----------
        v_f : float
            费米速度
        m0 : float
            质量参数

        Returns:
        --------
        np.ndarray
            (2n × 2n) Hamiltonian 矩阵
        """
        n = self.n_grid
        k_x = self.momentum_operator(0)

        # 质量项 (对角)
        k_vals = np.linspace(-np.pi, np.pi, n, endpoint=False)
        m_diag = m0 - k_vals**2

        # 组装 2×2 块矩阵
        H = np.zeros((2*n, 2*n), dtype=complex)

        # 对角块: m(k)·σ_z
        H[0:n, 0:n] = np.diag(m_diag)
        H[n:2*n, n:2*n] = np.diag(-m_diag)

        # 非对角块: v_F·k_x·σ_x
        H[0:n, n:2*n] = v_f * k_x
        H[n:2*n, 0:n] = v_f * k_x

        return H


def richardson_extrapolation(values: List[float], ratios: List[float],
                             order: int = 2) -> float:
    """
    Richardson 外推提高精度

    给定步长 h, h/r, h/r² 处的值 f_h, f_{h/r}, f_{h/r²}
    外推值 = f_{h/r} + (f_{h/r} - f_h) / (r^p - 1)

    Parameters:
    -----------
    values : List[float]
        不同步长下的计算值
    ratios : List[float]
        步长比
    order : int
        方法的阶数

    Returns:
    --------
    float
        外推后的高精度值
    """
    if len(values) < 2:
        return values[0]

    f1 = values[0]
    f2 = values[1]
    r = ratios[0]

    # 一次外推
    f_extrap = f2 + (f2 - f1) / (r**order - 1)
    return f_extrap
