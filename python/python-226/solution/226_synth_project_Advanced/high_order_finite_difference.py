"""
高阶有限差分算子与稳定性分析模块
===============================
实现中微子传播方程的空间离散化，提供2阶、4阶、6阶、8阶精度的有限差分算子。

核心数学公式：
1. 一阶导数（中心差分）：
   - 2阶精度: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
   - 4阶精度: f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)
   - 6阶精度: f'(x) ≈ [f(x+3h) - 9f(x+2h) + 45f(x+h) - 45f(x-h) + 9f(x-2h) - f(x-3h)] / (60h)

2. 二阶导数（中心差分）：
   - 2阶精度: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
   - 4阶精度: f''(x) ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / (12h²)

3. 截断误差分析（泰勒展开）：
   E_2nd = h²/6 f'''(ξ),  ξ ∈ [x-h, x+h]
   E_4th = h⁴/30 f⁽⁵⁾(ξ)
   E_6th = h⁶/140 f⁽⁷⁾(ξ)

4. von Neumann稳定性条件（FTCS格式）：
   对于 ∂u/∂t = -c ∂u/∂x，FTCS格式：
   u_j^{n+1} = u_j^n - (cΔt/(2Δx))(u_{j+1}^n - u_{j-1}^n)
   放大因子: G(k) = 1 - i(cΔt/Δx)sin(kΔx)
   |G|² = 1 + (cΔt/Δx)²sin²(kΔx) > 1  →  无条件不稳定

5. 改进格式（Lax-Wendroff / Leapfrog）：
   Lax-Wendroff: u_j^{n+1} = u_j^n - (ν/2)(u_{j+1}^n - u_{j-1}^n) + (ν²/2)(u_{j+1}^n - 2u_j^n + u_{j-1}^n)
   其中 ν = cΔt/Δx（CFL数）

6. 中微子传播方程的离散化：
   i dψ/dx = H(x)ψ  →  ψ(x+Δx) = [I - iΔx H(x)]ψ(x)（Euler格式）
   高阶格式: ψ(x+Δx) = exp(-iΔx H)ψ(x)（矩阵指数格式）

数据来源：
- 353_fd1d_advection_ftcs: FTCS格式的基础实现
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
from enum import Enum


class FDOrder(Enum):
    """有限差分精度阶数"""
    ORDER_2 = 2
    ORDER_4 = 4
    ORDER_6 = 6
    ORDER_8 = 8


class FDOperator:
    """
    有限差分算子类。

    实现任意阶精度的中心差分算子，包括：
    - 一阶导数
    - 二阶导数
    - 拉普拉斯算子
    - 对流算子

    核心方法基于泰勒展开的系数求解：
    Σ_k c_k f(x + k*h) = f^(n)(x) + O(h^p)
    """

    # 一阶导数中心差分系数
    FIRST_DERIV_COEFFS = {
        2: np.array([-0.5, 0.0, 0.5]),  # [-1, 0, 1] / (2h)
        4: np.array([1/12, -2/3, 0.0, 2/3, -1/12]) / 1,  # 4阶
        6: np.array([-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60]),  # 6阶
        8: np.array([1/280, -4/105, 1/5, -4/5, 0.0, 4/5, -1/5, 4/105, -1/280]),  # 8阶
    }

    # 二阶导数中心差分系数
    SECOND_DERIV_COEFFS = {
        2: np.array([1.0, -2.0, 1.0]),  # [1, -2, 1] / h²
        4: np.array([-1/12, 4/3, -5/2, 4/3, -1/12]),  # 4阶
        6: np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]),  # 6阶
        8: np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]),  # 8阶
    }

    def __init__(self, dx: float, order: int = 4, periodic: bool = False):
        """
        初始化有限差分算子。

        参数：
            dx: 网格间距
            order: 精度阶数 (2, 4, 6, 8)
            periodic: 是否使用周期边界条件

        边界处理：
            - 非周期边界使用单侧差分或镜像反射
            - 阶数自动裁剪到可用范围
        """
        if dx <= 0:
            raise ValueError(f"网格间距必须为正: dx={dx}")

        valid_orders = [2, 4, 6, 8]
        if order not in valid_orders:
            # 自动选择最近的可用阶数
            order = min(valid_orders, key=lambda x: abs(x - order))

        self.dx = dx
        self.order = FDOrder(order)
        self.periodic = periodic

        # 模板宽度
        self.stencil_width = order + 1
        self.half_width = order // 2

        # 系数
        self.first_deriv_coeffs = self.FIRST_DERIV_COEFFS[order]
        self.second_deriv_coeffs = self.SECOND_DERIV_COEFFS[order]

    def first_derivative(self, f: np.ndarray) -> np.ndarray:
        """
        计算一阶导数 f'(x)。

        实现公式：
        f'(x_j) ≈ Σ_k c_k f(x_{j+k}) / dx

        参数：
            f: 函数值数组 (N,)

        返回：
            df: 一阶导数数组 (N,)

        边界处理：
            - 周期边界: 使用模运算环绕
            - 非周期边界: 使用低阶单侧差分
        """
        N = len(f)
        df = np.zeros_like(f)
        coeffs = self.first_deriv_coeffs
        hw = self.half_width

        if self.periodic:
            # 周期边界：使用roll实现
            for k, c in enumerate(coeffs):
                shift = k - hw
                if abs(c) > 1e-15:
                    df += c * np.roll(f, -shift)
            df /= self.dx
        else:
            # 内部点：使用中心差分
            for j in range(hw, N - hw):
                for k, c in enumerate(coeffs):
                    df[j] += c * f[j + k - hw]
                df[j] /= self.dx

            # 边界点：使用低阶单侧差分
            if hw > 0:
                # 左边界：前向差分
                for j in range(min(hw, N)):
                    if j == 0:
                        df[j] = (f[1] - f[0]) / self.dx if N > 1 else 0.0
                    elif N > j + 1:
                        df[j] = (f[j+1] - f[j-1]) / (2 * self.dx)

                # 右边界：后向差分
                for j in range(max(N - hw, 0), N):
                    if j == N - 1:
                        df[j] = (f[-1] - f[-2]) / self.dx if N > 1 else 0.0
                    elif j > 0:
                        df[j] = (f[j+1] - f[j-1]) / (2 * self.dx)

        return df

    def second_derivative(self, f: np.ndarray) -> np.ndarray:
        """
        计算二阶导数 f''(x)。

        实现公式：
        f''(x_j) ≈ Σ_k c_k f(x_{j+k}) / dx²

        参数：
            f: 函数值数组 (N,)

        返回：
            d2f: 二阶导数数组 (N,)
        """
        N = len(f)
        d2f = np.zeros_like(f)
        coeffs = self.second_deriv_coeffs
        hw = self.half_width

        if self.periodic:
            for k, c in enumerate(coeffs):
                shift = k - hw
                if abs(c) > 1e-15:
                    d2f += c * np.roll(f, -shift)
            d2f /= self.dx ** 2
        else:
            for j in range(hw, N - hw):
                for k, c in enumerate(coeffs):
                    d2f[j] += c * f[j + k - hw]
                d2f[j] /= self.dx ** 2

            # 边界处理
            if hw > 0 and N > 2:
                for j in range(min(hw, N)):
                    if j == 0 and N > 2:
                        d2f[j] = (f[2] - 2*f[1] + f[0]) / self.dx**2
                    elif N > 2:
                        d2f[j] = (f[j+1] - 2*f[j] + f[j-1]) / self.dx**2

                for j in range(max(N - hw, 0), N):
                    if j == N - 1 and N > 2:
                        d2f[j] = (f[-1] - 2*f[-2] + f[-3]) / self.dx**2
                    elif N > 2:
                        d2f[j] = (f[j+1] - 2*f[j] + f[j-1]) / self.dx**2

        return d2f

    def laplacian_matrix(self, N: int) -> np.ndarray:
        """
        构造拉普拉斯算子的稀疏矩阵表示。

        数学表达：
        L = (1/dx²) × tridiag(1, -2, 1)  [2阶精度]

        对于高阶精度，使用更宽的带状矩阵。

        参数：
            N: 网格点数

        返回：
            L: N×N拉普拉斯矩阵
        """
        coeffs = self.second_deriv_coeffs
        hw = self.half_width
        L = np.zeros((N, N))

        for j in range(N):
            for k, c in enumerate(coeffs):
                col = j + k - hw
                if 0 <= col < N:
                    L[j, col] = c / self.dx ** 2
                elif self.periodic:
                    L[j, col % N] += c / self.dx ** 2

        return L

    def advection_matrix(self, c: float, N: int) -> np.ndarray:
        """
        构造对流算子 c × d/dx 的矩阵表示。

        物理方程：
        ∂u/∂t = -c ∂u/∂x

        离散化：
        du_j/dt = -c × Σ_k a_k u_{j+k} / dx

        参数：
            c: 对流速度
            N: 网格点数

        返回：
            A: N×N对流矩阵
        """
        coeffs = self.first_deriv_coeffs
        hw = self.half_width
        A = np.zeros((N, N))

        for j in range(N):
            for k, a in enumerate(coeffs):
                col = j + k - hw
                if 0 <= col < N:
                    A[j, col] = -c * a / self.dx
                elif self.periodic:
                    A[j, col % N] += -c * a / self.dx

        return A

    def truncation_error_estimate(self, f_exact: np.ndarray, f_numerical: np.ndarray) -> float:
        """
        估计截断误差。

        误差公式：
        E_p = ||f_exact - f_numerical||_∞

        对于p阶精度：E_p ~ O(h^p)

        参数：
            f_exact: 精确解
            f_numerical: 数值解

        返回：
            error: L∞误差
        """
        return np.max(np.abs(f_exact - f_numerical))


class StabilityAnalyzer:
    """
    von Neumann稳定性分析器。

    核心方法：
    1. 计算放大因子 G(k) = ψ^{n+1}(k) / ψ^n(k)
    2. 稳定性条件: |G(k)| ≤ 1 对所有波数 k
    3. CFL条件: ν = cΔt/Δx ≤ ν_max

    对于中微子传播方程 i dψ/dx = Hψ：
    放大矩阵: G = I - iΔx H
    稳定性: ρ(G) ≤ 1 (谱半径)
    """

    def __init__(self, dx: float, dt: float):
        """
        初始化稳定性分析器。

        参数：
            dx: 空间步长
            dt: 时间步长
        """
        self.dx = dx
        self.dt = dt
        self.cfl = dt / dx  # CFL数（c=1时）

    def ftcs_amplification_factor(self, c: float, k: np.ndarray) -> np.ndarray:
        """
        FTCS格式的放大因子。

        数学推导：
        对于 ∂u/∂t = -c ∂u/∂x，FTCS格式：
        u_j^{n+1} = u_j^n - (cΔt/(2Δx))(u_{j+1}^n - u_{j-1}^n)

        代入 u_j^n = G^n e^{ijkΔx}：
        G = 1 - i(cΔt/Δx)sin(kΔx)
        |G|² = 1 + (cΔt/Δx)²sin²(kΔx) > 1

        结论：FTCS对纯对流方程无条件不稳定！

        参数：
            c: 对流速度
            k: 波数数组

        返回：
            G: 放大因子数组（复数）
        """
        nu = c * self.dt / self.dx  # CFL数
        G = 1.0 - 1j * nu * np.sin(k * self.dx)
        return G

    def lax_wendroff_amplification(self, c: float, k: np.ndarray) -> np.ndarray:
        """
        Lax-Wendroff格式的放大因子。

        数学推导：
        u_j^{n+1} = u_j^n - (ν/2)(u_{j+1} - u_{j-1}) + (ν²/2)(u_{j+1} - 2u_j + u_{j-1})

        G = 1 - iν sin(kΔx) + ν²(cos(kΔx) - 1)
        |G|² = 1 - ν²(1-ν²)(1-cos(kΔx))² ≤ 1  当 |ν| ≤ 1

        参数：
            c: 对流速度
            k: 波数数组

        返回：
            G: 放大因子数组
        """
        nu = c * self.dt / self.dx
        theta = k * self.dx
        G = 1.0 - 1j * nu * np.sin(theta) + nu**2 * (np.cos(theta) - 1.0)
        return G

    def leapfrog_amplification(self, c: float, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Leapfrog格式的放大因子（两个分支）。

        数学推导：
        u_j^{n+1} = u_j^{n-1} - ν(u_{j+1}^n - u_{j-1}^n)

        特征方程: G² + 2iν sin(θ) G - 1 = 0
        G = -iν sin(θ) ± √(1 - ν²sin²(θ))

        稳定性条件: |ν| ≤ 1

        参数：
            c: 对流速度
            k: 波数数组

        返回：
            G_plus, G_minus: 两个放大因子分支
        """
        nu = c * self.dt / self.dx
        theta = k * self.dx

        discriminant = 1.0 - nu**2 * np.sin(theta)**2
        discriminant = np.maximum(discriminant, 0)  # 防止负值

        sqrt_disc = np.sqrt(discriminant + 0j)
        G_plus = -1j * nu * np.sin(theta) + sqrt_disc
        G_minus = -1j * nu * np.sin(theta) - sqrt_disc

        return G_plus, G_minus

    def neutrino_evolution_stability(self, H: np.ndarray, dx: float) -> dict:
        """
        分析中微子演化方程的稳定性。

        物理方程：
        i dψ/dx = H(x)ψ

        Euler格式：ψ(x+Δx) = (I - iΔx H)ψ(x)
        放大矩阵：G = I - iΔx H

        稳定性条件：ρ(G) ≤ 1

        对于厄米特矩阵H，本征值为实数λ_j：
        G_j = 1 - iΔx λ_j
        |G_j|² = 1 + (Δx λ_j)² > 1

        结论：显式Euler格式对薛定谔方程不稳定！
        需要使用酉演化算符：G = exp(-iΔx H)

        参数：
            H: 哈密顿量矩阵 (N×N)
            dx: 传播步长

        返回：
            stability_info: 稳定性分析结果字典
        """
        # 计算H的本征值
        eigenvalues = np.linalg.eigvalsh(H)

        # Euler格式的放大因子
        G_euler = np.eye(len(H)) - 1j * dx * H
        eig_G_euler = np.linalg.eigvals(G_euler)

        # 矩阵指数格式的放大因子
        from scipy.linalg import expm
        G_exact = expm(-1j * dx * H)
        eig_G_exact = np.linalg.eigvals(G_exact)

        # 分析结果
        result = {
            'H_eigenvalues': eigenvalues,
            'euler_spectral_radius': np.max(np.abs(eig_G_euler)),
            'euler_is_stable': np.max(np.abs(eig_G_euler)) <= 1.0 + 1e-10,
            'exact_spectral_radius': np.max(np.abs(eig_G_exact)),
            'exact_is_unitary': np.allclose(np.abs(eig_G_exact), 1.0),
            'max_eigenvalue': np.max(np.abs(eigenvalues)),
            'critical_dx': np.pi / np.max(np.abs(eigenvalues)) if np.max(np.abs(eigenvalues)) > 0 else np.inf,
        }

        return result

    def cfl_condition(self, c: float, max_order: int = 4) -> float:
        """
        计算CFL条件限制的最大时间步长。

        CFL条件：
        ν = cΔt/Δx ≤ ν_max

        对于不同格式：
        - FTCS: 无条件不稳定（ν_max = 0）
        - Lax-Wendroff: ν_max = 1
        - Leapfrog: ν_max = 1
        - 高阶格式: ν_max 取决于具体格式

        参数：
            c: 对流速度
            max_order: 空间差分精度阶数

        返回：
            dt_max: 最大允许时间步长
        """
        # 对于中微子传播，使用矩阵指数方法无CFL限制
        # 对于有限差分格式，CFL数约为1
        nu_max = 1.0 / (max_order / 2)  # 高阶格式更严格

        dt_max = nu_max * self.dx / abs(c) if c != 0 else np.inf

        return dt_max


def convergence_test(fd_op: FDOperator, f_func: Callable, df_exact: Callable,
                     N_values: List[int] = None) -> dict:
    """
    收敛性测试：验证有限差分的精度阶数。

    方法：
    1. 在不同分辨率下计算数值导数
    2. 计算与精确解的误差
    3. 拟合误差 vs dx 的幂律关系: E ~ C × dx^p
    4. 提取实际精度阶数 p

    参数：
        fd_op: 有限差分算子
        f_func: 测试函数 f(x)
        df_exact: 精确导数函数 f'(x)
        N_values: 网格点数列表

    返回：
        results: 收敛性测试结果
    """
    if N_values is None:
        N_values = [50, 100, 200, 400, 800]

    errors = []
    dx_values = []

    for N in N_values:
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        dx = x[1] - x[0]
        dx_values.append(dx)

        f = f_func(x)
        df_num = fd_op.first_derivative(f)
        df_ex = df_exact(x)

        error = np.max(np.abs(df_num - df_ex))
        errors.append(error)

    errors = np.array(errors)
    dx_values = np.array(dx_values)

    # 拟合 log(E) = p × log(dx) + log(C)
    mask = errors > 0
    if np.sum(mask) >= 2:
        log_dx = np.log(dx_values[mask])
        log_err = np.log(errors[mask])
        p_fit, C_fit = np.polyfit(log_dx, log_err, 1)
        observed_order = -p_fit
    else:
        observed_order = 0

    return {
        'N_values': N_values,
        'dx_values': dx_values,
        'errors': errors,
        'observed_order': observed_order,
        'expected_order': fd_op.order.value,
        'convergence_ok': abs(observed_order - fd_op.order.value) < 0.5,
    }
