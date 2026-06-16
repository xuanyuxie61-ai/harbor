"""
qgp_stability_analysis.py — von Neumann 稳定性分析与 CFL 条件
===============================================================

融合种子项目:
  - 737_matrix_analyze (矩阵属性分析)
  - 898_polynomials (多项式分析)

本模块对 WENO5 + TVD-RK3 数值格式进行 von Neumann 稳定性分析,
并检验矩阵属性以确保数值格式的良态性.

von Neumann 稳定性分析:
-----------------------

将数值解 Fourier 分解:
    u_j^n = hat{u}^n * exp(i * k * j * dx)

代入离散化格式, 得到放大因子:
    hat{u}^{n+1} = G(k * dx) * hat{u}^n

稳定性条件:
    |G(xi)| <= 1 + O(dt)  for all xi = k*dx in [-pi, pi]

对于 TVD-RK3 + WENO5 + LF:
    G(xi) 是 dt, dx, 特征速度 alpha 的函数.

简化分析 (线性 advection + 中心差分):
    du/dt + a * du/dx = 0
    WENO5 线性部分 ~ 五阶中心差分
    amplification: G(xi) = 1 - i*cfl*sin(xi)*(1 + correction terms)

矩阵分析 (源自 737_matrix_analyze):
    构造离散化算子矩阵 A:
        du/dt = A * u
    检验:
    - A 的特征值: Re(lambda) <= 0 (稳定性)
    - 谱半径 rho(A): dt < 2/rho(A) (显式格式稳定)
    - 条件数 cond(A): 反映数值刚性
    - 对角优势: 保证迭代收敛
    - 对称性: 影响特征值实数性
    - 正规性: A*A^T = A^T*A (影响特征向量正交性)
"""

import numpy as np
from qgp_config import NumericalParams
from qgp_grid import QGPGrid
from qgp_eos import QGPEquationOfState


class StabilityAnalyzer:
    """
    数值格式稳定性分析器

    执行:
    1. von Neumann 放大因子计算
    2. 离散化算子矩阵特征值分析
    3. 矩阵属性检验 (源自 737_matrix_analyze)
    4. CFL 条件验证
    """

    def __init__(self, grid: QGPGrid, eos: QGPEquationOfState):
        """
        Args:
            grid: 计算网格
            eos: 状态方程
        """
        self.grid = grid
        self.eos = eos

    def von_neumann_amplification(self, wave_numbers: np.ndarray = None,
                                    dt: float = 0.01,
                                    advection_speed: float = 0.5,
                                    viscosity: float = 0.0) -> dict:
        """
        计算 von Neumann 放大因子 G(xi)

        模型方程: du/dt + a*du/dx = nu*d^2u/dx^2
        (线性 advection + 扩散)

        WENO5 线性部分 (均匀权重时的等价格式):
            五阶中心差分:
            (du/dx)_j ≈ (-u_{j-2} + 8*u_{j-1} - 8*u_{j+1} + u_{j+2}) / (12*dx)

        放大因子:
            G(xi) = 1 - dt/dx * (a * i * A(xi) + nu/dx * B(xi))

        其中 A(xi) 为五阶差分的 Fourier 符号:
            A(xi) = (-e^{-2i*xi} + 8*e^{-i*xi} - 8*e^{i*xi} + e^{2i*xi}) / 12
                  = (8*sin(xi) - sin(2*xi)) / 6

        B(xi) 为二阶 Laplacian 的 Fourier 符号:
            B(xi) = 2*cos(xi) - 2

        稳定条件: |G(xi)| <= 1 for all xi in [-pi, pi]

        Args:
            wave_numbers: 波数数组 xi = k*dx (默认 [-pi, pi])
            dt: 时间步长
            advection_speed: 对流速度 a
            viscosity: 扩散系数 nu

        Returns:
            {'xi': array, 'G': array, '|G|': array,
             'stable': bool, 'max_amplification': float}
        """
        if wave_numbers is None:
            xi = np.linspace(-np.pi, np.pi, 500)
        else:
            xi = wave_numbers

        dx = self.grid.dx

        # CFL 数
        cfl = abs(advection_speed) * dt / dx
        nu_cfl = viscosity * dt / dx**2

        # 五阶中心差分 Fourier 符号
        A_xi = (8.0 * np.sin(xi) - np.sin(2.0 * xi)) / 6.0

        # 二阶 Laplacian Fourier 符号
        B_xi = 2.0 * np.cos(xi) - 2.0

        # 放大因子
        # G = 1 - cfl * i * A(xi) + nu_cfl * B(xi)
        G = 1.0 - cfl * 1j * A_xi + nu_cfl * B_xi

        # TVD-RK3 修正 (对线性格式)
        # TVD-RK3: G_rk3 = (1/3) + (2/3)*G_1 * (1 + G_1/2 + G_1^2/4)
        # 简化: 用 Euler 放大因子 G_1 的三次多项式近似
        G_euler = G
        # TVD-RK3 放大因子 (线性化)
        G_rk3 = (1.0/3.0) + (2.0/3.0) * (
            0.75 + 0.25 * G_euler + 0.25 * G_euler**2
        ) * G_euler

        G_abs = np.abs(G_rk3)
        max_amp = np.max(G_abs)
        is_stable = max_amp <= 1.0 + 1.0e-10  # 允许小量误差

        return {
            'xi': xi,
            'G': G_rk3,
            'G_abs': G_abs,
            'stable': is_stable,
            'max_amplification': float(max_amp),
            'cfl': cfl,
            'nu_cfl': nu_cfl,
        }

    def dispersion_relation(self, wave_numbers: np.ndarray = None) -> dict:
        """
        数值色散关系分析

        精确色散: omega = a * k (线性 advection)
        数值色散: omega_num = -ln(G) / (i * dt)

        数值耗散: Re(omega_num) != a * k
        数值色散: Im(omega_num) != 0

        物理波速: v_phase = Re(omega) / k
        群速度: v_group = d(Re(omega)) / dk

        Args:
            wave_numbers: 波数数组

        Returns:
            {'xi': array, 'phase_velocity_ratio': array,
             'dissipation_rate': array}
        """
        result = self.von_neumann_amplification(wave_numbers)
        xi = result['xi']
        G = result['G']
        dt = NumericalParams.DT_MAX

        # 数值频率
        # G = exp(-i * omega * dt) => omega = i * ln(G) / dt
        G_safe = np.where(np.abs(G) > 1.0e-15, G, 1.0e-15)
        omega_num = 1j * np.log(G_safe + 0j) / dt

        # 相速度比 (数值 / 精确)
        k = xi / self.grid.dx
        k_safe = np.where(np.abs(k) > 1.0e-15, k, 1.0e-15)
        phase_vel = np.real(omega_num) / k_safe
        # 归一化 (精确值为 1)
        phase_ratio = np.abs(phase_vel)

        # 耗散率
        dissipation = -np.imag(omega_num)

        return {
            'xi': xi,
            'phase_velocity_ratio': phase_ratio,
            'dissipation_rate': dissipation,
            'group_velocity': np.gradient(np.real(omega_num), xi) / self.grid.dx,
        }

    def build_discretization_matrix(self, advection_speed: float = 0.5) -> np.ndarray:
        """
        构造一维空间离散化矩阵 A

        对于 du/dt + a*du/dx = 0, 使用五阶中心差分:
            A_{j,j-2} = a / (12*dx)
            A_{j,j-1} = -8a / (12*dx)
            A_{j,j+1} = 8a / (12*dx)
            A_{j,j+2} = -a / (12*dx)

        (周期性边界条件)

        Args:
            advection_speed: 对流速度

        Returns:
            N x N 离散化矩阵
        """
        N = self.grid.nx
        dx = self.grid.dx
        a = advection_speed

        A = np.zeros((N, N))
        coeff = a / (12.0 * dx)

        for j in range(N):
            A[j, (j-2) % N] = coeff       # j-2
            A[j, (j-1) % N] = -8.0 * coeff  # j-1
            A[j, (j+1) % N] = 8.0 * coeff   # j+1
            A[j, (j+2) % N] = -coeff        # j+2

        return A

    def matrix_eigenvalue_analysis(self, A: np.ndarray = None) -> dict:
        """
        矩阵特征值分析 (源自 737_matrix_analyze)

        分析离散化算子矩阵的谱性质:
        1. 特征值分布 (谱半径)
        2. 条件数
        3. 对称性检验
        4. 对角优势检验

        稳定性条件:
            所有特征值实部 <= 0 (或 |lambda_max * dt| < 稳定域)

        对于中心差分 advection 算子:
            特征值为纯虚数 (反 Hermitian)
            => 需要耗散 (人工粘滞或迎风) 来稳定

        Args:
            A: 离散化矩阵 (若为 None, 自动构建)

        Returns:
            {'eigenvalues': complex_array,
             'spectral_radius': float,
             'condition_number': float,
             'is_symmetric': bool,
             'is_diagonally_dominant': bool,
             'max_real_part': float}
        """
        if A is None:
            A = self.build_discretization_matrix()

        # 特征值
        eigenvalues = np.linalg.eigvals(A)

        # 谱半径
        rho = np.max(np.abs(eigenvalues))

        # 条件数 (2-范数)
        try:
            cond = np.linalg.cond(A)
        except np.linalg.LinAlgError:
            cond = np.inf

        # 对称性: ||A - A^T||_F
        symmetry_norm = np.linalg.norm(A - A.T, 'fro')
        is_symmetric = symmetry_norm < 1.0e-10 * np.linalg.norm(A, 'fro')

        # 对角优势
        diag_abs = np.abs(np.diag(A))
        off_diag_sum = np.sum(np.abs(A), axis=1) - diag_abs
        is_diag_dominant = bool(np.all(diag_abs >= off_diag_sum - 1.0e-10))

        # 最大实部
        max_real = np.max(np.real(eigenvalues))

        # 正规性: ||A*A^H - A^H*A||_F
        AH = A.conj().T
        normality = np.linalg.norm(A @ AH - AH @ A, 'fro')

        return {
            'eigenvalues': eigenvalues,
            'spectral_radius': float(rho),
            'condition_number': float(cond),
            'is_symmetric': bool(is_symmetric),
            'is_diagonally_dominant': is_diag_dominant,
            'max_real_part': float(max_real),
            'normality_measure': float(normality),
            'symmetry_norm': float(symmetry_norm),
        }

    def cfl_analysis(self, advection_speed: float = 0.5,
                      sound_speed: float = 0.577) -> dict:
        """
        CFL 条件分析

        CFL 条件: dt <= CFL_number * dx / (|v| + c_s)

        对于 QGP 流体力学:
            特征速度 = |v| + c_s (流速 + 声速)
            典型值: |v| ~ 0.5-0.8, c_s ~ 1/sqrt(3) ~ 0.577

        分析不同网格分辨率下的最大 dt:
            dt_max(N) = CFL * L / (N * v_char)

        Args:
            advection_speed: 典型流速
            sound_speed: 声速

        Returns:
            {'dt_max': float, 'v_char': float, 'cfl_number': float,
             'grid_resolution_limit': float}
        """
        v_char = advection_speed + sound_speed
        dx = self.grid.dx
        dy = self.grid.dy
        dh = min(dx, dy)

        cfl = NumericalParams.CFL_NUMBER
        dt_max = cfl * dh / v_char

        # 不同分辨率下的 dt_max
        resolutions = [20, 40, 80, 160, 320]
        dt_by_resolution = {}
        L = self.grid.x_int[-1] - self.grid.x_int[0]
        for N in resolutions:
            dx_N = L / N
            dt_by_resolution[N] = cfl * dx_N / v_char

        return {
            'dt_max': float(dt_max),
            'v_char': float(v_char),
            'cfl_number': cfl,
            'dh': float(dh),
            'dt_by_resolution': dt_by_resolution,
        }

    def convergence_test(self, orders: list = [1, 2, 3, 5]) -> dict:
        """
        数值收敛阶测试

        方法: 在逐步加密的网格上求解同一问题,
        测量误差与网格间距的关系:
            error ~ C * dx^p
        => log(error) = log(C) + p * log(dx)
        => p = d(log(error)) / d(log(dx))

        对于:
            一阶 upwind: p = 1
            二阶中心: p = 2
            TVD-RK3: p = 3 (时间)
            WENO5: p = 5 (空间, 光滑区域)

        Args:
            orders: 期望的收敛阶列表

        Returns:
            {'dx_values': array, 'errors': array,
             'observed_order': float, 'expected_order': int}
        """
        # 使用模型问题: du/dt + a*du/dx = 0
        # 精确解: u(x,t) = sin(2*pi*(x - a*t) / L)
        a = 0.5
        L = self.grid.x_int[-1] - self.grid.x_int[0]
        T = 0.5  # 演化时间

        def exact_solution(x, t):
            return np.sin(2.0 * np.pi * (x - a * t) / L)

        # 不同分辨率
        N_values = [20, 40, 80, 160]
        errors = []
        dx_values = []

        for N in N_values:
            dx = L / N
            x = np.linspace(0, L, N, endpoint=False)
            dt = 0.4 * dx / abs(a)  # 满足 CFL
            n_steps = int(T / dt)

            # 一阶 upwind 求解
            u = exact_solution(x, 0.0)
            for _ in range(n_steps):
                u_new = u.copy()
                for j in range(N):
                    u_jm1 = u[(j-1) % N]
                    if a > 0:
                        u_new[j] = u[j] - a * dt / dx * (u[j] - u_jm1)
                    else:
                        u_jp1 = u[(j+1) % N]
                        u_new[j] = u[j] - a * dt / dx * (u_jp1 - u[j])
                u = u_new

            u_exact = exact_solution(x, T)
            error = np.sqrt(np.mean((u - u_exact)**2))
            errors.append(error)
            dx_values.append(dx)

        # 计算观测收敛阶
        dx_arr = np.array(dx_values)
        err_arr = np.array(errors)
        if len(dx_arr) >= 2 and np.all(err_arr > 0):
            log_dx = np.log(dx_arr)
            log_err = np.log(err_arr)
            observed_order = np.abs(np.polyfit(log_dx, log_err, 1)[0])
        else:
            observed_order = 1.0

        return {
            'dx_values': dx_arr,
            'errors': err_arr,
            'observed_order': float(observed_order),
            'N_values': N_values,
        }

    def full_stability_report(self) -> dict:
        """
        生成完整稳定性分析报告

        Returns:
            包含所有分析结果的字典
        """
        # 1. von Neumann 分析
        vn = self.von_neumann_amplification()

        # 2. 矩阵特征值分析
        A = self.build_discretization_matrix()
        eig = self.matrix_eigenvalue_analysis(A)

        # 3. CFL 分析
        cfl = self.cfl_analysis()

        # 4. 色散关系
        disp = self.dispersion_relation()

        # 5. 收敛阶测试
        conv = self.convergence_test()

        return {
            'von_neumann': vn,
            'eigenvalue_analysis': {
                k: v for k, v in eig.items() if k != 'eigenvalues'
            },
            'cfl': cfl,
            'dispersion': {
                'max_phase_error': float(np.max(np.abs(
                    disp['phase_velocity_ratio'] - 1.0))),
                'max_dissipation': float(np.max(disp['dissipation_rate'])),
            },
            'convergence': {
                'observed_order': conv['observed_order'],
                'errors': conv['errors'].tolist(),
            },
        }
