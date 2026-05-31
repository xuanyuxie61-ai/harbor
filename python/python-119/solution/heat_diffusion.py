"""
heat_diffusion.py
热传导与温度场分析模块

融合原项目:
- 360_fd1d_heat_explicit: 1D 显式热方程有限差分
- 404_fem2d_heat_rectangle: 2D 有限元热传导（三角形网格、二次基函数）
- 877_poisson_2d: 2D 泊松方程 Jacobi 迭代求解

功能:
1. 1D 显式有限差分求解聚合物薄膜沿厚度方向的温度分布
2. 2D 有限元求解基底-薄膜界面的热传导
3. Jacobi 迭代求解稳态热平衡（融合 877_poisson_2d）
4. 热扩散系数与玻璃化转变的关联
"""

import numpy as np
from typing import Tuple, Callable, Optional
from numeric_utils import safe_divide


class HeatDiffusion1D:
    """
    1D 热传导求解器（显式有限差分）。
    
    融合原项目 360_fd1d_heat_explicit:
        求解方程:
            ∂T/∂t = α ∂²T/∂x² + Q(x,t)
        
        边界条件:
            T(0,t) = T_left(t), T(L,t) = T_right(t)
        
        显式格式:
            T_i^{n+1} = T_i^n + CFL * (T_{i-1}^n - 2T_i^n + T_{i+1}^n) + dt * Q_i
        
        CFL = α dt / dx² < 0.5 保证稳定性
    
    物理应用:
        模拟聚合物薄膜在淬火过程中的温度梯度演化。
        玻璃化转变前沿的传播速度 v ≈ sqrt(α/τ) 与热扩散系数 α 相关。
    """
    
    def __init__(
        self,
        L: float = 10.0,
        nx: int = 101,
        alpha: float = 0.1,
        dt: float = 0.001,
    ):
        """
        参数:
            L: 薄膜厚度
            nx: 空间格点数
            alpha: 热扩散系数
            dt: 时间步长
        """
        if L <= 0 or nx < 3 or alpha <= 0 or dt <= 0:
            raise ValueError("参数必须满足: L>0, nx>=3, alpha>0, dt>0")
        
        self.L = L
        self.nx = nx
        self.alpha = alpha
        self.dt = dt
        self.dx = L / (nx - 1)
        
        # CFL 检查
        self.cfl = alpha * dt / (self.dx ** 2)
        if self.cfl >= 0.5:
            # 自动调整 dt 以满足稳定性
            self.dt = 0.45 * (self.dx ** 2) / alpha
            self.cfl = alpha * self.dt / (self.dx ** 2)
        
        self.x = np.linspace(0, L, nx)
        self.T = np.ones(nx)
    
    def solve_step(
        self,
        T_left: float,
        T_right: float,
        heat_source: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        执行一个时间步的热扩散。
        
        参数:
            T_left: 左边界温度
            T_right: 右边界温度
            heat_source: (nx,) 热源项 Q(x)
        
        返回:
            更新后的温度分布
        """
        T_new = np.zeros_like(self.T)
        
        # 内部点显式更新
        for i in range(1, self.nx - 1):
            diffusion = self.cfl * (self.T[i-1] - 2.0 * self.T[i] + self.T[i+1])
            source = 0.0
            if heat_source is not None:
                source = self.dt * heat_source[i]
            T_new[i] = self.T[i] + diffusion + source
        
        # Dirichlet 边界条件
        T_new[0] = T_left
        T_new[-1] = T_right
        
        self.T = T_new
        return self.T.copy()
    
    def solve_steady(
        self,
        T_left: float,
        T_right: float,
        heat_source: Optional[np.ndarray] = None,
        max_iter: int = 10000,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """
        求解稳态温度分布。
        
        参数:
            T_left, T_right: 边界温度
            heat_source: 热源项
            max_iter: 最大迭代次数
            tol: 收敛容差
        
        返回:
            稳态温度分布
        """
        self.T = np.linspace(T_left, T_right, self.nx)
        
        for it in range(max_iter):
            T_old = self.T.copy()
            self.solve_step(T_left, T_right, heat_source)
            
            diff = np.max(np.abs(self.T - T_old))
            if diff < tol:
                break
        
        return self.T.copy()
    
    def thermal_gradient(self) -> np.ndarray:
        """
        计算温度梯度 dT/dx（中心差分）。
        
        返回:
            (nx,) 梯度数组
        """
        grad = np.zeros(self.nx)
        grad[1:-1] = (self.T[2:] - self.T[:-2]) / (2.0 * self.dx)
        grad[0] = (self.T[1] - self.T[0]) / self.dx
        grad[-1] = (self.T[-1] - self.T[-2]) / self.dx
        return grad


class HeatDiffusion2DFEM:
    """
    2D 热传导有限元求解器（简化版）。
    
    融合原项目 404_fem2d_heat_rectangle:
        在矩形区域 [0,Lx]×[0,Ly] 上求解:
            ∂T/∂t - α ∇²T = Q(x,y,t)
        
        使用三角形网格和向后 Euler 时间离散:
            (M + dt K) T^{n+1} = M T^n + dt F^{n+1}
        
        其中:
            M: 质量矩阵
            K: 刚度矩阵
            F: 载荷向量
    
    物理应用:
        模拟聚合物薄膜-基底系统的二维热传导，
        评估界面处的温度梯度对玻璃化转变的影响。
    """
    
    def __init__(
        self,
        Lx: float = 10.0,
        Ly: float = 10.0,
        nx: int = 21,
        ny: int = 21,
        alpha: float = 0.1,
    ):
        """
        参数:
            Lx, Ly: 矩形区域尺寸
            nx, ny: x, y 方向格点数
            alpha: 热扩散系数
        """
        if nx < 2 or ny < 2:
            raise ValueError("nx, ny 必须 >= 2")
        
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.alpha = alpha
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # 节点坐标
        self.X = np.zeros((nx, ny))
        self.Y = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                self.X[i, j] = i * self.dx
                self.Y[i, j] = j * self.dy
        
        # 温度场
        self.T = np.ones((nx, ny))
    
    def _laplacian_5point(self, T: np.ndarray) -> np.ndarray:
        """
        5 点 Laplacian 离散。
        
        公式:
            ∇²T ≈ (T_{i-1,j} + T_{i+1,j} + T_{i,j-1} + T_{i,j+1} - 4T_{i,j}) / h²
        
        参数:
            T: (nx, ny) 温度场
        
        返回:
            (nx, ny) Laplacian
        """
        lap = np.zeros_like(T)
        
        # 内部点
        lap[1:-1, 1:-1] = (
            (T[:-2, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[2:, 1:-1]) / (self.dx ** 2)
            + (T[1:-1, :-2] - 2.0 * T[1:-1, 1:-1] + T[1:-1, 2:]) / (self.dy ** 2)
        )
        
        return lap
    
    def solve_backward_euler_step(
        self,
        dt: float,
        boundary_T: Callable[[np.ndarray, np.ndarray], np.ndarray],
        heat_source: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        执行一步向后 Euler 时间积分。
        
        融合 404_fem2d_heat_rectangle 的时间离散思想:
            (I - dt α ∇²) T^{n+1} = T^n + dt Q
        
        使用 Jacobi 迭代求解隐式方程。
        
        参数:
            dt: 时间步长
            boundary_T: 边界温度函数 boundary_T(X, Y) -> T_boundary
            heat_source: (nx, ny) 热源项
        
        返回:
            更新后的温度场
        """
        if dt <= 0:
            raise ValueError("dt 必须 > 0")
        
        # 右端项
        rhs = self.T.copy()
        if heat_source is not None:
            rhs += dt * heat_source
        
        # Jacobi 迭代求解 (I - dt α ∇²) T_new = rhs
        T_new = self.T.copy()
        
        # CFL-like 约束用于 Jacobi 收敛
        cfl_x = self.alpha * dt / (self.dx ** 2)
        cfl_y = self.alpha * dt / (self.dy ** 2)
        
        if cfl_x + cfl_y > 0.5:
            # 减小 dt 确保 Jacobi 收敛
            dt = 0.45 / (self.alpha * (1.0 / self.dx ** 2 + 1.0 / self.dy ** 2))
        
        max_iter = 5000
        tol = 1e-8
        
        for it in range(max_iter):
            T_old = T_new.copy()
            
            # Jacobi 更新: T_new = rhs + dt * alpha * laplacian(T_old)
            lap = self._laplacian_5point(T_old)
            T_new = rhs + dt * self.alpha * lap
            
            # 应用边界条件
            T_boundary = boundary_T(self.X, self.Y)
            T_new[0, :] = T_boundary[0, :]
            T_new[-1, :] = T_boundary[-1, :]
            T_new[:, 0] = T_boundary[:, 0]
            T_new[:, -1] = T_boundary[:, -1]
            
            diff = np.max(np.abs(T_new - T_old))
            if diff < tol:
                break
        
        self.T = T_new
        return self.T.copy()
    
    def solve_steady_jacobi(
        self,
        boundary_T: Callable[[np.ndarray, np.ndarray], np.ndarray],
        heat_source: Optional[np.ndarray] = None,
        max_iter: int = 10000,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """
        使用 Jacobi 迭代求解稳态热传导方程。
        
        融合原项目 877_poisson_2d:
            求解 ∇²T = -Q/α
        
        Jacobi 迭代:
            T_{i,j}^{new} = (T_{i-1,j} + T_{i+1,j} + T_{i,j-1} + T_{i,j+1} + dx² Q/α) / 4
        
        参数:
            boundary_T: 边界温度函数
            heat_source: 热源项
            max_iter: 最大迭代次数
            tol: 收敛容差
        
        返回:
            稳态温度场
        """
        self.T = np.ones((self.nx, self.ny))
        
        # 初始化边界
        T_boundary = boundary_T(self.X, self.Y)
        self.T[0, :] = T_boundary[0, :]
        self.T[-1, :] = T_boundary[-1, :]
        self.T[:, 0] = T_boundary[:, 0]
        self.T[:, -1] = T_boundary[:, -1]
        
        # 内部点初始插值
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                self.T[i, j] = (
                    self.T[i, 0] * (self.ny - 1 - j) / (self.ny - 1)
                    + self.T[i, -1] * j / (self.ny - 1)
                ) * 0.5 + (
                    self.T[0, j] * (self.nx - 1 - i) / (self.nx - 1)
                    + self.T[-1, j] * i / (self.nx - 1)
                ) * 0.5
        
        source_term = np.zeros((self.nx, self.ny))
        if heat_source is not None:
            source_term = heat_source * (self.dx ** 2) / self.alpha
        
        for it in range(max_iter):
            T_old = self.T.copy()
            
            # Jacobi  sweep（融合 877_poisson_2d 的 jacobi 函数）
            # 内部点更新
            self.T[1:-1, 1:-1] = 0.25 * (
                T_old[:-2, 1:-1]
                + T_old[2:, 1:-1]
                + T_old[1:-1, :-2]
                + T_old[1:-1, 2:]
                + source_term[1:-1, 1:-1]
            )
            
            # 边界保持不变
            self.T[0, :] = T_boundary[0, :]
            self.T[-1, :] = T_boundary[-1, :]
            self.T[:, 0] = T_boundary[:, 0]
            self.T[:, -1] = T_boundary[:, -1]
            
            diff = np.max(np.abs(self.T - T_old))
            if diff < tol:
                break
        
        return self.T.copy()
    
    def effective_thermal_conductivity(self) -> float:
        """
        估算有效热导率（基于温度梯度与热流的关系）。
        
        公式:
            κ_eff = -q / (∂T/∂x) = α * ρ * C_p
        
        这里使用简化估计:
            κ_eff ≈ α * <T>
        
        返回:
            有效热导率
        """
        return float(self.alpha * np.mean(self.T))
