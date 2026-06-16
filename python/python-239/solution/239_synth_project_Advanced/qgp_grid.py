"""
qgp_grid.py — 二维横向网格与坐标系统
======================================

融合种子项目: 115_box_games (网格/棋盘结构),
             654_lattice_rule (格点结构)

本模块构建 QGP 流体动力学模拟的二维横向计算网格.

坐标系统:
---------
采用 Milne 坐标 (tau, x, y, eta_s):
    t = tau * cosh(eta_s)
    z = tau * sinh(eta_s)

其中 tau = sqrt(t^2 - z^2) 为固有时间,
     eta_s = 0.5 * ln((t+z)/(t-z)) 为时空快度.

在 eta_s = 0 (中间快度) 截面, 流体动力学方程简化为 2+1 维:
    partial_mu T^{mu nu} = 0,  mu, nu = {tau, x, y}

度量张量:
    g_{mu nu} = diag(1, -1, -1, -tau^2)  (Milne 坐标)

边界条件:
    x 方向: 周期性 (横向环面近似)
    y 方向: 外推出边界 (自由流边界)

网格结构 (源自 115_box_games):
    二维数组 (ny+2*ng, nx+2*ng), 包含鬼单元
    鬼单元用于施加边界条件, 宽度 = WENO5 模板半径 r=3

格点积分 (源自 654_lattice_rule):
    用于相空间积分的 Fibonacci 格点和标准格点
"""

import numpy as np
from qgp_config import NumericalParams


class QGPGrid:
    """
    二维横向计算网格

    构建 (x, y) 平面上的均匀矩形网格, 带鬼单元边界.

    Attributes:
        nx, ny: 内部网格点数
        ng: 鬼单元宽度
        dx, dy: 网格间距 (fm)
        x, y: 坐标数组 (含鬼单元)
        x_int, y_int: 内部区域坐标
        X, Y: 二维坐标网格 (含鬼单元)
        X_int, Y_int: 内部区域坐标网格
        R: 径向坐标 r = sqrt(x^2 + y^2)
    """

    def __init__(self, nx: int = None, ny: int = None,
                 x_range: tuple = None, y_range: tuple = None,
                 ghost_cells: int = None):
        """
        初始化网格

        Args:
            nx: x 方向网格点数
            ny: y 方向网格点数
            x_range: (x_min, x_max) in fm
            y_range: (y_min, y_max) in fm
            ghost_cells: 鬼单元宽度
        """
        params = NumericalParams
        self.nx = nx if nx is not None else params.NX
        self.ny = ny if ny is not None else params.NY
        self.ng = ghost_cells if ghost_cells is not None else params.GHOST_CELLS

        x_min, x_max = x_range if x_range else (params.X_MIN, params.X_MAX)
        y_min, y_max = y_range if y_range else (params.Y_MIN, params.Y_MAX)

        # 网格间距 (dx = L / N, 周期性边界)
        self.dx = (x_max - x_min) / self.nx
        self.dy = (y_max - y_min) / self.ny

        # 全网格坐标 (含鬼单元)
        self.x = np.linspace(x_min - self.ng * self.dx,
                             x_max + (self.ng - 1) * self.dx,
                             self.nx + 2 * self.ng)
        self.y = np.linspace(y_min - self.ng * self.dy,
                             y_max + (self.ng - 1) * self.dy,
                             self.ny + 2 * self.ng)

        # 内部区域坐标
        self.x_int = self.x[self.ng:self.ng + self.nx]
        self.y_int = self.y[self.ng:self.ng + self.ny]

        # 二维网格
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        self.X_int = self.X[self.ng:self.ng + self.ny,
                            self.ng:self.ng + self.nx]
        self.Y_int = self.Y[self.ng:self.ng + self.ny,
                            self.ng:self.ng + self.nx]

        # 径向坐标
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.R_int = np.sqrt(self.X_int**2 + self.Y_int**2)

        # 方位角 (用于流谐波分析)
        self.PHI = np.arctan2(self.Y, self.X)

        # 面积元
        self.dV = self.dx * self.dy

        # 预计算格点积分节点 (用于 Cooper-Frye 积分)
        self._init_lattice_points()

    def _init_lattice_points(self):
        """
        初始化格点积分节点 (Fibonacci 格点)

        2D Fibonacci 格点:
            N = F_k (第 k 个 Fibonacci 数)
            生成向量: z = [1, F_{k-1}]
            节点: x_j = (j * z / N) mod 1, j = 0, ..., N-1

        Fibonacci 格点具有最优的 2D 均匀分布性质,
        其星偏差 D_N ~ O(1/N), 远优于随机 Monte Carlo 的 O(1/sqrt(N)).
        """
        # 计算 Fibonacci 数列
        fib = [1, 1]
        while fib[-1] < NumericalParams.LATTICE_POINTS:
            fib.append(fib[-1] + fib[-2])
        self.n_fib = fib[-1]
        self.fib_prev = fib[-2]

        # Fibonacci 格点 (单位正方形)
        j = np.arange(self.n_fib, dtype=np.float64)
        self.lattice_u = np.mod(j / self.n_fib, 1.0)
        self.lattice_v = np.mod(j * self.fib_prev / self.n_fib, 1.0)

    def apply_periodic_bc_x(self, field: np.ndarray) -> np.ndarray:
        """
        施加 x 方向周期性边界条件

        周期性边界: f(x + L) = f(x)
        物理: 模拟横向平面上的环面拓扑

        实现:
            ghost_left[i] = field[ny+ng-i, ...]  (环绕)
            ghost_right[i] = field[i+ng, ...]

        Args:
            field: 二维场 (ny+2*ng, nx+2*ng)

        Returns:
            更新后的场 (鬼单元已填充)
        """
        f = field.copy()
        for i in range(self.ng):
            # 左边界
            f[:, i] = f[:, self.nx + i]
            # 右边界
            f[:, self.ng + self.nx + i] = f[:, self.ng + i]
        return f

    def apply_outflow_bc_y(self, field: np.ndarray) -> np.ndarray:
        """
        施加 y 方向外推出边界条件

        外推边界: df/dy = 0 (零梯度)
        物理: 允许扰动自由传出计算区域, 无反射

        实现: 线性外推
            ghost[j] = 2*f[boundary] - f[interior]

        Args:
            field: 二维场

        Returns:
            更新后的场
        """
        f = field.copy()
        for i in range(self.ng):
            # 下边界: 线性外推
            f[i, :] = 2.0 * f[self.ng, :] - f[2 * self.ng - i, :]
            # 上边界
            f[self.ng + self.ny + i, :] = (
                2.0 * f[self.ng + self.ny - 1, :] -
                f[self.ng + self.ny - 2 - i, :]
            )
        return f

    def apply_boundary_conditions(self, field: np.ndarray) -> np.ndarray:
        """
        施加组合边界条件

        x: 周期性  |  y: 外推

        Args:
            field: 二维场

        Returns:
            更新后的场
        """
        f = self.apply_periodic_bc_x(field)
        f = self.apply_outflow_bc_y(f)
        # 再次应用 x 周期性 (因 y 外推可能影响角落)
        f = self.apply_periodic_bc_x(f)
        return f

    def apply_bc_conserved(self, D: np.ndarray,
                           Sx: np.ndarray, Sy: np.ndarray) -> tuple:
        """
        对守恒变量组施加边界条件

        Args:
            D: 修正重子密度
            Sx, Sy: 动量密度

        Returns:
            (D, Sx, Sy) 鬼单元已填充
        """
        D = self.apply_boundary_conditions(D)
        Sx = self.apply_boundary_conditions(Sx)
        Sy = self.apply_boundary_conditions(Sy)
        return D, Sx, Sy

    def interior_slice(self, field: np.ndarray) -> np.ndarray:
        """
        提取内部区域 (去除鬼单元)

        Args:
            field: 完整网格上的场

        Returns:
            (ny, nx) 内部区域
        """
        return field[self.ng:self.ng + self.ny,
                     self.ng:self.ng + self.nx]

    def gradient_magnitude(self, field: np.ndarray) -> np.ndarray:
        """
        计算场梯度的模 |grad f|

        使用中心差分:
            (df/dx)_i = (f_{i+1} - f_{i-1}) / (2*dx)

        Args:
            field: 二维场 (含鬼单元)

        Returns:
            |grad f| 在内部区域
        """
        # 先填充鬼单元
        f = self.apply_boundary_conditions(field)
        dfdx = (f[self.ng:self.ng + self.ny, self.ng + 1:self.ng + self.nx + 1] -
                f[self.ng:self.ng + self.ny, self.ng - 1:self.ng + self.nx - 1]) / (2.0 * self.dx)
        dfdy = (f[self.ng + 1:self.ng + self.ny + 1, self.ng:self.ng + self.nx] -
                f[self.ng - 1:self.ng + self.ny - 1, self.ng:self.ng + self.nx]) / (2.0 * self.dy)
        return np.sqrt(dfdx**2 + dfdy**2)

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        计算二维 Laplace 算子

        nabla^2 f = d^2f/dx^2 + d^2f/dy^2
        ≈ (f_{i+1,j} - 2*f_{i,j} + f_{i-1,j}) / dx^2
        + (f_{i,j+1} - 2*f_{i,j} + f_{i,j-1}) / dy^2

        Args:
            field: 二维场 (含鬼单元)

        Returns:
            nabla^2 f 在内部区域
        """
        f = self.apply_boundary_conditions(field)
        ng = self.ng
        d2fdx2 = (f[ng:ng + self.ny, ng + 1:ng + self.nx + 1] -
                   2.0 * f[ng:ng + self.ny, ng:ng + self.nx] +
                   f[ng:ng + self.ny, ng - 1:ng + self.nx - 1]) / self.dx**2
        d2fdy2 = (f[ng + 1:ng + self.ny + 1, ng:ng + self.nx] -
                   2.0 * f[ng:ng + self.ny, ng:ng + self.nx] +
                   f[ng - 1:ng + self.ny - 1, ng:ng + self.nx]) / self.dy**2
        return d2fdx2 + d2fdy2

    def integrate_2d(self, field_int: np.ndarray) -> float:
        """
        二维积分 (矩形法则)

        Integral f dx dy ≈ sum f_{i,j} * dx * dy

        Args:
            field_int: 内部区域上的场 (ny, nx)

        Returns:
            积分值
        """
        return np.sum(field_int) * self.dV

    def lattice_integrate_2d(self, field_func, x_range: tuple,
                              y_range: tuple) -> float:
        """
        使用 Fibonacci 格点的准 Monte Carlo 积分

        I = (b-a)*(d-c) * (1/N) * sum_{j=0}^{N-1} f(x_j, y_j)

        其中 (x_j, y_j) 为 Fibonacci 格点映射到积分区域.

        Args:
            field_func: 被积函数 f(x, y)
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)

        Returns:
            积分近似值
        """
        xa, xb = x_range
        ya, yb = y_range
        x_pts = xa + (xb - xa) * self.lattice_u
        y_pts = ya + (yb - ya) * self.lattice_v
        f_vals = field_func(x_pts, y_pts)
        area = (xb - xa) * (yb - ya)
        return area * np.mean(f_vals)

    def total_points(self) -> int:
        """返回总网格点数 (含鬼单元)"""
        return (self.ny + 2 * self.ng) * (self.nx + 2 * self.ng)

    def internal_points(self) -> int:
        """返回内部网格点数"""
        return self.nx * self.ny

    def aspect_ratio(self) -> float:
        """网格纵横比"""
        Lx = self.nx * self.dx
        Ly = self.ny * self.dy
        return Lx / max(Ly, 1.0e-15)

    def cfl_time_step(self, max_velocity: float, sound_speed: float) -> float:
        """
        计算满足 CFL 条件的最大时间步长

        CFL 条件:
            dt <= CFL * min(dx, dy) / (|v|_max + c_s)

        其中 |v|_max 为最大流速, c_s 为声速.
        信息传播速度 = 流速 + 声速 (特征速度).

        Args:
            max_velocity: 最大流速 (自然单位)
            sound_speed: 声速 c_s

        Returns:
            最大时间步长 dt (fm/c)
        """
        v_max = max(max_velocity, 1.0e-10)
        char_speed = v_max + sound_speed
        dh = min(self.dx, self.dy)
        dt_cfl = NumericalParams.CFL_NUMBER * dh / char_speed
        dt_cfl = min(dt_cfl, NumericalParams.DT_MAX)
        dt_cfl = max(dt_cfl, NumericalParams.DT_MIN)
        return dt_cfl
