"""
finite_diff_schemes.py — 高阶有限差分格式与稳定性分析
======================================================

融合种子项目:
  - 437_flame_ode : 火焰ODE中高阶差分算子的构造
  - 086_biharmonic_cheby1d : Chebyshev 谱微分矩阵

本模块实现用于离散化 Rossi-Greisen 级联方程的高阶有限差分格式,
包括:
  1. 中心差分 (2阶, 4阶, 6阶, 8阶)
  2. 迎风差分 (1阶, 3阶, 5阶 — 用于对流占优区域)
  3. 紧致差分 (Pade 格式 — 四阶精度三对角)
  4. Chebyshev 谱微分矩阵
  5. von Neumann 稳定性分析
  6. 矩阵耗散/色散关系计算

核心数学框架:
  对于微分算子 d^n f / dx^n 在均匀网格 x_j = j*h 上:

  2阶中心差分:
    D2 f_j = (f_{j+1} - 2*f_j + f_{j-1}) / h^2
    修正波数: k_hat^2 = (2 - 2*cos(k*h)) / h^2

  4阶中心差分 (一阶导数):
    D4 f_j = (-f_{j+2} + 8*f_{j+1} - 8*f_{j-1} + f_{j-2}) / (12*h)
    修正波数: k_hat = (8*sin(k*h) - sin(2*k*h)) / (6*h)

  紧致 Pade 格式 (四阶三对角):
    alpha * f'_{j-1} + f'_j + alpha * f'_{j+1}
      = a * (f_{j+1} - f_{j-1}) / (2*h)
    其中 alpha = 1/4, a = 3/2 给出六阶精度

  von Neumann 稳定性条件:
    对于 du/dt = A * u 的半离散格式,
    稳定域要求: dt * max|eigenvalues(A)| <= C_stable
    其中 C_stable 取决于时间推进方法

  CFL 条件:
    dt <= C_CFL * h / max|characteristic_speed|
    对于抛物型: dt <= C * h^2 / diffusion_coefficient
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StabilityResult:
    """稳定性分析结果"""
    scheme_name: str
    order: int
    max_dt_stable: float          # 最大稳定时间步长
    spectral_radius: float        # 谱半径 (最大特征值模)
    cfl_number: float             # CFL 数
    is_stable: bool
    amplification_factors: List[complex]  # 放大因子
    dissipation_error: List[float]        # 耗散误差
    dispersion_error: List[float]         # 色散误差


class FiniteDiffOperator:
    """
    有限差分算子构造器

    支持任意阶有限差分格式的系统生成。
    使用 Taylor 展开匹配法求系数:
      sum_i c_i * f(x + i*h) = f'(x) + O(h^p)

    其中 p 为精度阶数。
    """

    def __init__(self, n_points: int):
        self.n = n_points
        self._stencil_cache = {}

    def central_first_derivative(self, order: int = 2) -> List[float]:
        """
        中心差分一阶导数格式

        2阶:  [-1/2, 0, 1/2] / h
        4阶:  [1/12, -2/3, 0, 2/3, -1/12] / h
        6阶:  [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60] / h
        8阶:  [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280] / h

        修正波数分析:
          k_hat * h = sum_s c_s * sin(s * k * h)
        """
        if order == 2:
            return [-0.5, 0.0, 0.5]
        elif order == 4:
            return [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]
        elif order == 6:
            return [
                -1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0,
                0.0,
                3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0,
            ]
        elif order == 8:
            return [
                1.0 / 280.0, -4.0 / 105.0, 1.0 / 5.0, -4.0 / 5.0,
                0.0,
                4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0,
            ]
        else:
            # 一般阶数: 使用 Lagrange 插值求系数
            return self._compute_central_coefficients(1, order)

    def central_second_derivative(self, order: int = 2) -> List[float]:
        """
        中心差分二阶导数格式

        2阶: [1, -2, 1] / h^2
        4阶: [-1/12, 4/3, -5/2, 4/3, -1/12] / h^2
        6阶: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] / h^2
        """
        if order == 2:
            return [1.0, -2.0, 1.0]
        elif order == 4:
            return [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
        elif order == 6:
            return [
                1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0,
                -49.0 / 18.0,
                3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0,
            ]
        else:
            return self._compute_central_coefficients(2, order)

    def upwind_first_derivative(self, order: int = 1) -> List[float]:
        """
        迎风差分一阶导数 (适用于对流占优)

        1阶迎风:  [-1, 1] / h  (向后差分)
        3阶迎风:  [1/3, -3/2, 3, -11/6] / h  (三点向后 + 一点前)
        5阶迎风:  [-1/5, 1/4, -1, 2, -25/12, 83/60] / h

        修正波数 (1阶迎风):
          k_hat * h = 1 - exp(-i*k*h)
          实部 (耗散): 1 - cos(k*h)  >= 0  (正耗散, 稳定)
          虚部 (色散): sin(k*h)       (无色散误差至 O(kh))
        """
        if order == 1:
            return [-1.0, 1.0]
        elif order == 3:
            return [1.0 / 3.0, -3.0 / 2.0, 3.0, -11.0 / 6.0]
        elif order == 5:
            return [
                -1.0 / 5.0, 1.0 / 4.0, -1.0, 2.0,
                -25.0 / 12.0, 83.0 / 60.0,
            ]
        else:
            return [-1.0, 1.0]

    def compact_pade_first(self, alpha: float = 0.25) -> Tuple[List[float], List[float]]:
        """
        紧致 Pade 格式 (三对角隐式差分)

        alpha * f'_{j-1} + f'_j + alpha * f'_{j+1}
            = a * (f_{j+1} - f_{j-1}) / (2*h)

        对于 alpha = 1/4:
          左端:  [1/4, 1, 1/4]
          右端:  [3/4, 0, -3/4] / (2*h)  (a = 3/2)
          精度: O(h^4)

        对于 alpha = 1/3:
          a = 14/9, b = 1/9
          右端: [1/9, 14/9, 0, -14/9, -1/9] / (2*h)  (五对角)
          精度: O(h^6)

        谱特性:
          k_hat * h = (a * sin(k*h)) / (1 + 2*alpha*cos(k*h))
        """
        a = 1.5  # 对应 alpha = 1/4
        lhs = [alpha, 1.0, alpha]
        rhs = [a / (2.0), 0.0, -a / (2.0)]
        return lhs, rhs

    def modified_wavenumber(
        self, stencil: List[float], h: float, derivative_order: int,
        n_kh: int = 200,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        计算修正波数 (modified wavenumber) 分析

        对于格式 sum_j c_j * f(x + j*h),
        修正波数 k_hat 满足:
          sum_j c_j * exp(i*j*k*h) = i^p * (k_hat * h)^p

        返回:
          kh_values: k*h 的值 [0, pi]
          khat_real: 修正波数实部 (色散特性)
          khat_imag: 修正波数虚部 (耗散特性)
        """
        half_len = len(stencil) // 2
        kh_values = []
        khat_real = []
        khat_imag = []

        for ik in range(n_kh + 1):
            kh = math.pi * ik / n_kh
            # 计算 sum_j c_j * exp(i*j*kh)
            real_sum = 0.0
            imag_sum = 0.0
            for j, c in enumerate(stencil):
                offset = j - half_len
                angle = offset * kh
                real_sum += c * math.cos(angle)
                imag_sum += c * math.sin(angle)

            # 提取修正波数
            if derivative_order == 1:
                # i * khat * h = real_sum + i * imag_sum
                khat_r = imag_sum
                khat_i = -real_sum
            elif derivative_order == 2:
                # -khat^2 * h^2 = real_sum + i * imag_sum
                khat_r = math.sqrt(max(-real_sum, 0.0))
                khat_i = imag_sum
            else:
                khat_r = imag_sum
                khat_i = -real_sum

            kh_values.append(kh)
            khat_real.append(khat_r)
            khat_imag.append(khat_i)

        return kh_values, khat_real, khat_imag

    def _compute_central_coefficients(
        self, deriv_order: int, accuracy_order: int,
    ) -> List[float]:
        """
        使用 Taylor 展开匹配计算中心差分系数

        求解线性系统: V * c = e_d
        其中 V 为 Vandermonde 矩阵:
          V[i][j] = j^i / i!
        """
        n_stencil = accuracy_order + deriv_order
        if n_stencil % 2 == 0:
            n_stencil += 1  # 中心差分需要奇数个点
        half = n_stencil // 2

        # 构建 Vandermonde 系统
        # 点: -half, -(half-1), ..., 0, ..., half
        points = list(range(-half, half + 1))
        n = len(points)

        # 矩阵 A[i][j] = points[j]^i
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] = points[j] ** i

        # 右端向量: e_d = [0, 0, ..., d!, ..., 0]
        b = [0.0] * n
        b[deriv_order] = math.factorial(deriv_order)

        # 高斯消元求解
        coeffs = self._solve_linear_system(A, b)
        return coeffs

    def _solve_linear_system(
        self, A: List[List[float]], b: List[float],
    ) -> List[float]:
        """高斯消元法求解线性方程组 (带部分主元)"""
        n = len(b)
        # 增广矩阵
        M = [row[:] + [bi] for row, bi in zip(A, b)]

        for col in range(n):
            # 部分主元选取
            max_val = abs(M[col][col])
            max_row = col
            for row in range(col + 1, n):
                if abs(M[row][col]) > max_val:
                    max_val = abs(M[row][col])
                    max_row = row
            if max_val < 1e-15:
                continue
            M[col], M[max_row] = M[max_row], M[col]

            # 消元
            pivot = M[col][col]
            for row in range(col + 1, n):
                factor = M[row][col] / pivot
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]

        # 回代
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(M[i][i]) < 1e-15:
                x[i] = 0.0
                continue
            s = M[i][n]
            for j in range(i + 1, n):
                s -= M[i][j] * x[j]
            x[i] = s / M[i][i]
        return x


class StabilityAnalyzer:
    """
    von Neumann 稳定性分析器

    对于半离散格式 du/dt = L * u,
    其中 L 为空间离散算子的矩阵表示,

    稳定性条件:
      显式 Euler:  |1 + dt * lambda| <= 1  (对所有特征值 lambda)
      RK4:         |R(dt * lambda)| <= 1
        其中 R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24

    对于扩散方程 du/dt = D * d^2u/dx^2:
      显式 Euler: dt <= h^2 / (2*D)
      RK4:        dt <= 2.785 * h^2 / (2*D)  (放宽 ~2.8 倍)
    """

    def __init__(self, fd_operator: Optional[FiniteDiffOperator] = None):
        self.fd = fd_operator or FiniteDiffOperator(64)

    def analyze_diffusion_stability(
        self,
        diffusion_coeff: float,
        grid_spacing: float,
        scheme_order: int = 2,
        time_scheme: str = "euler",
    ) -> StabilityResult:
        """
        分析扩散方程的稳定性

        du/dt = D * d^2u/dx^2

        空间离散后: du_i/dt = D * L_ij * u_j
        其中 L 为二阶差分矩阵

        特征值分析:
          lambda_k = -4*D/h^2 * sin^2(k*h/2)  (2阶中心差分)
          |lambda_max| = 4*D/h^2

        稳定性限制:
          Euler: dt <= h^2 / (2*D)
          RK2:   dt <= h^2 / D
          RK4:   dt <= 2.785 * h^2 / (2*D)
        """
        D = diffusion_coeff
        h = grid_spacing

        # 最大特征值 (二阶差分)
        if scheme_order == 2:
            lambda_max = 4.0 * D / (h * h)
        elif scheme_order == 4:
            # 四阶差分的谱半径更大
            lambda_max = D / (h * h) * (
                -(-5.0 / 2.0) + 2.0 * 4.0 / 3.0 + 2.0 / 12.0
            )
            lambda_max = abs(lambda_max) * 2.0
        else:
            lambda_max = 4.0 * D / (h * h) * (1.0 + 0.1 * (scheme_order - 2))

        # 时间步长限制
        if time_scheme == "euler":
            max_dt = 1.0 / lambda_max
            stability_limit = 0.5  # 安全系数
        elif time_scheme == "rk2":
            max_dt = 2.0 / lambda_max
            stability_limit = 1.0
        elif time_scheme == "rk4":
            max_dt = 2.785 / lambda_max
            stability_limit = 2.785
        else:
            max_dt = 1.0 / lambda_max
            stability_limit = 0.5

        max_dt_stable = max_dt * stability_limit
        cfl = D * max_dt_stable / (h * h)

        # 放大因子 (von Neumann 分析)
        n_modes = 64
        amp_factors = []
        diss_errors = []
        disp_errors = []

        for ik in range(n_modes + 1):
            kh = math.pi * ik / n_modes
            # 修正波数
            if scheme_order == 2:
                khat_sq_h2 = 2.0 - 2.0 * math.cos(kh)
            else:
                # 4阶近似
                khat_sq_h2 = (
                    2.0 - 2.0 * math.cos(kh)
                    + (2.0 * math.cos(kh) - 2.0 * math.cos(2 * kh)) / 12.0
                )

            z = -D * max_dt_stable * khat_sq_h2 / (h * h)

            if time_scheme == "euler":
                g = 1.0 + z
            elif time_scheme == "rk4":
                g = 1.0 + z + z * z / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0
            else:
                g = 1.0 + z + z * z / 2.0

            amp_factors.append(complex(g, 0))
            exact_g = math.exp(z)
            diss_errors.append(abs(abs(g) - abs(exact_g)))
            disp_errors.append(abs(math.atan2(0, g) - 0))

        return StabilityResult(
            scheme_name=f"diffusion_{time_scheme}_o{scheme_order}",
            order=scheme_order,
            max_dt_stable=max_dt_stable,
            spectral_radius=lambda_max,
            cfl_number=cfl,
            is_stable=max_dt_stable > 0,
            amplification_factors=amp_factors,
            dissipation_error=diss_errors,
            dispersion_error=disp_errors,
        )

    def analyze_cascade_stability(
        self,
        sigma_a: float,       # 光子吸收截面 [1/X0]
        sigma_p: float,       # 对产生截面 [1/X0]
        sigma_b: float,       # bremsstrahlung 截面 [1/X0]
        grid_spacing: float,  # 深度网格间距 [X0]
        fd_order: int = 4,
    ) -> StabilityResult:
        """
        分析 Rossi-Greisen 级联方程的稳定性

        级联方程离散后的矩阵形式:
          d/dt [phi_e, phi_gamma]^T = M * [phi_e, phi_gamma]^T

        其中 M 为:
          M = [[-(sigma_b + sigma_ion), sigma_p],
               [sigma_a,              -sigma_a    ]]

        特征值:
          lambda = trace(M)/2 ± sqrt(trace(M)^2/4 - det(M))

        稳定性要求:
          Re(lambda) <= 0  (物理耗散)
          dt * |lambda| <= C_stable  (数值稳定)
        """
        # 构建耦合矩阵
        trace_M = -(sigma_b + 0.1) - sigma_a  # 对角线之和
        det_M = (sigma_b + 0.1) * sigma_a - sigma_p * sigma_a

        discriminant = trace_M * trace_M / 4.0 - det_M

        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            lambda1 = trace_M / 2.0 + sqrt_disc
            lambda2 = trace_M / 2.0 - sqrt_disc
        else:
            sqrt_disc = math.sqrt(-discriminant)
            lambda1_real = trace_M / 2.0
            lambda2_real = trace_M / 2.0
            lambda1 = lambda1_real  # 取实部做稳定性判断
            lambda2 = lambda2_real

        spectral_radius = max(abs(lambda1), abs(lambda2))

        # 稳定时间步长 (RK4)
        max_dt = 2.785 / max(spectral_radius, 1e-30)
        max_dt = min(max_dt, grid_spacing * 0.5)  # 不超过网格间距的一半

        cfl = spectral_radius * max_dt

        # 生成放大因子
        amp_factors = []
        diss_errors = []
        disp_errors = []

        for ik in range(33):
            frac = ik / 32.0
            z = -spectral_radius * max_dt * frac
            g = 1.0 + z + z * z / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0
            amp_factors.append(complex(g, 0))
            exact_g = math.exp(z)
            diss_errors.append(abs(abs(g) - abs(exact_g)))
            disp_errors.append(0.0)

        return StabilityResult(
            scheme_name="cascade_rk4",
            order=fd_order,
            max_dt_stable=max_dt,
            spectral_radius=spectral_radius,
            cfl_number=cfl,
            is_stable=max_dt > 0 and (lambda1 <= 0 or abs(lambda1) < 1e-10),
            amplification_factors=amp_factors,
            dissipation_error=diss_errors,
            dispersion_error=disp_errors,
        )

    def compute_spectral_radius(
        self, matrix: List[List[float]], n_power: int = 100,
    ) -> float:
        """
        幂迭代法计算矩阵谱半径

        rho(A) = max |lambda_i(A)|

        算法:
          v_0 = random unit vector
          for k = 1, 2, ...:
            w = A * v_{k-1}
            v_k = w / ||w||
            lambda_est = v_k^T * A * v_k
        """
        n = len(matrix)
        if n == 0:
            return 0.0

        # 初始向量
        v = [1.0 / math.sqrt(n)] * n

        for _ in range(n_power):
            # 矩阵-向量乘法
            w = [0.0] * n
            for i in range(n):
                for j in range(n):
                    w[i] += matrix[i][j] * v[j]

            # 归一化
            norm_w = math.sqrt(sum(x * x for x in w))
            if norm_w < 1e-30:
                return 0.0
            v = [x / norm_w for x in w]

        # 计算 Rayleigh 商
        Av = [0.0] * n
        for i in range(n):
            for j in range(n):
                Av[i] += matrix[i][j] * v[j]
        rayleigh = sum(v[i] * Av[i] for i in range(n))

        return abs(rayleigh)
