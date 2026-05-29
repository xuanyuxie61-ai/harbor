"""
reaction_diffusion_kernel.py
基于项目 487_gray_scott_pde 与 353_fd1d_advection_ftcs 的
反应扩散动力学驱动的量子特征映射模块。

核心数学模型:
1. Gray-Scott 反应扩散方程 (Pattern Formation):
   dU/dt = D_u * nabla^2 U - U*V^2 + gamma*(1 - U)
   dV/dt = D_v * nabla^2 V + U*V^2 - (gamma + kappa)*V

2. 9点高阶 Laplacian (Mehrstellenverfahren, O(h^4) 精度):
   nabla^2 A ≈ (1*A_{i-1,j-1} + 4*A_{i-1,j} + 1*A_{i-1,j+1}
              + 4*A_{i,j-1} - 20*A_{i,j} + 4*A_{i,j+1}
              + 1*A_{i+1,j-1} + 4*A_{i+1,j} + 1*A_{i+1,j+1}) / (6*dx^2)

3. 一维对流方程 (用于量子态输运模拟):
   du/dt + c * du/dx = 0
   FTCS 格式: u_i^{n+1} = u_i^n - c*dt/(2*dx)*(u_{i+1}^n - u_{i-1}^n)
   (已知不稳定，仅用于教学对比与 Trotter 误差分析)

4. 量子特征映射:
   将反应扩散产生的时空模式编码为量子电路参数 theta(x,t)。
"""

import numpy as np
from typing import Tuple, Optional


def laplacian9_torus(field: np.ndarray, dx: float) -> np.ndarray:
    """
    9点高阶 Laplacian 算子，周期边界条件 (Torus topology)。
    模板:
        [ 1  4  1 ]
        [ 4 -20  4 ] / (6 * dx^2)
        [ 1  4  1 ]
    精度: O(h^4)
    """
    if field.ndim != 2:
        raise ValueError("Field must be 2D array")
    if dx <= 0:
        raise ValueError("dx must be positive")
    ny, nx = field.shape
    if nx < 3 or ny < 3:
        raise ValueError("Field dimensions must be at least 3x3")

    result = np.zeros_like(field)
    # 周期边界索引
    for i in range(ny):
        im1 = (i - 1) % ny
        ip1 = (i + 1) % ny
        for j in range(nx):
            jm1 = (j - 1) % nx
            jp1 = (j + 1) % nx
            result[i, j] = (
                1.0 * field[im1, jm1] + 4.0 * field[im1, j] + 1.0 * field[im1, jp1]
                + 4.0 * field[i, jm1] - 20.0 * field[i, j] + 4.0 * field[i, jp1]
                + 1.0 * field[ip1, jm1] + 4.0 * field[ip1, j] + 1.0 * field[ip1, jp1]
            ) / (6.0 * dx * dx)
    return result


def gray_scott_step(
    U: np.ndarray,
    V: np.ndarray,
    dt: float,
    dx: float,
    D_u: float,
    D_v: float,
    gamma: float,
    kappa: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行一步 Gray-Scott 反应扩散方程的显式 Euler 时间推进。
    边界: 周期边界 (通过 laplacian9_torus 实现)。

    稳定性约束 (von Neumann 分析):
    dt <= dx^2 / (4 * max(D_u, D_v))
    """
    if dt <= 0 or dx <= 0:
        raise ValueError("dt and dx must be positive")
    if D_u < 0 or D_v < 0:
        raise ValueError("Diffusion coefficients must be non-negative")

    # 稳定性检查
    stability_limit = dx * dx / (4.0 * max(D_u, D_v) + 1e-15)
    if dt > stability_limit:
        # 自动调整 dt 以满足稳定性
        dt = 0.5 * stability_limit

    lapU = laplacian9_torus(U, dx)
    lapV = laplacian9_torus(V, dx)

    # 反应项
    reaction = U * V * V

    dUdt = D_u * lapU - reaction + gamma * (1.0 - U)
    dVdt = D_v * lapV + reaction - (gamma + kappa) * V

    U_new = U + dt * dUdt
    V_new = V + dt * dVdt

    # 物理约束: 浓度非负
    U_new = np.clip(U_new, 0.0, 1.0)
    V_new = np.clip(V_new, 0.0, 1.0)

    return U_new, V_new


def gray_scott_simulation(
    nx: int = 64,
    ny: int = 64,
    n_steps: int = 5000,
    D_u: float = 8.0e-5,
    D_v: float = 4.0e-5,
    gamma: float = 0.024,
    kappa: float = 0.06,
    dt: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    运行 Gray-Scott 反应扩散模拟。
    初始条件: V 在中心区域为正弦平方分布，U = 1 - 2V。
    返回最终的 U, V 浓度场。
    """
    if nx < 3 or ny < 3:
        raise ValueError("Grid dimensions must be >= 3")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    dx = 2.5 / (nx - 1)
    dy = 2.5 / (ny - 1)
    dx = min(dx, dy)  # 使用统一空间步长

    if dt is None:
        dt = 0.5 * dx * dx / (4.0 * max(D_u, D_v) + 1e-15)

    # 初始条件
    x = np.linspace(0.0, 2.5, nx)
    y = np.linspace(0.0, 2.5, ny)
    X, Y = np.meshgrid(x, y)

    V = np.zeros((ny, nx))
    mask = (X >= 1.0) & (X <= 1.5) & (Y >= 1.0) & (Y <= 1.5)
    V[mask] = 0.25 * (np.sin(4.0 * np.pi * X[mask]) ** 2) * (np.sin(4.0 * np.pi * Y[mask]) ** 2)
    U = 1.0 - 2.0 * V

    # 时间推进
    for _ in range(n_steps):
        U, V = gray_scott_step(U, V, dt, dx, D_u, D_v, gamma, kappa)

    return U, V


def advection_ftcs_step(
    u: np.ndarray,
    c: float,
    dt: float,
    dx: float
) -> np.ndarray:
    """
    一维对流方程的 FTCS (Forward Time, Centered Space) 单步推进。
    du/dt + c * du/dx = 0
    注意: FTCS 对于纯对流问题是无条件不稳定的 (|G| > 1)。
    本函数仅用于 Trotter 误差分析与教学目的。

    更新公式:
    u_i^{n+1} = u_i^n - c*dt/(2*dx) * (u_{i+1}^n - u_{i-1}^n)
    """
    # TODO: Implement the FTCS scheme for 1D advection equation.
    # The update formula is:
    #   u_i^{n+1} = u_i^n - c*dt/(2*dx) * (u_{i+1}^n - u_{i-1}^n)
    # Use periodic boundary conditions.
    # Validate inputs and raise appropriate errors.
    pass


def pattern_to_quantum_parameters(
    pattern: np.ndarray,
    n_qubits: int,
    n_layers: int
) -> np.ndarray:
    """
    将反应扩散模式映射为量子电路参数。
    数学映射:
    theta_{k,l} = pi * (pattern_idx / max_pattern) - pi/2
    其中 pattern 的值被线性插值到参数空间 [-pi/2, pi/2]。

    参数:
        pattern: 2D 反应扩散模式
        n_qubits: 量子比特数
        n_layers: 电路层数
    返回:
        params: 形状为 (n_layers, n_qubits) 的参数数组
    """
    if pattern.size == 0:
        raise ValueError("Pattern must not be empty")
    if n_qubits <= 0 or n_layers <= 0:
        raise ValueError("n_qubits and n_layers must be positive")

    # 将 pattern 展平并归一化
    flat = pattern.flatten()
    p_min, p_max = flat.min(), flat.max()
    if abs(p_max - p_min) < 1e-15:
        # 常数场，添加微小扰动
        flat = flat + 1e-8 * np.sin(np.arange(len(flat)))
        p_min, p_max = flat.min(), flat.max()

    normalized = (flat - p_min) / (p_max - p_min)

    # 插值到所需参数数量
    n_params = n_layers * n_qubits
    indices = np.linspace(0, len(normalized) - 1, n_params)
    idx_low = np.floor(indices).astype(int)
    idx_high = np.minimum(idx_low + 1, len(normalized) - 1)
    frac = indices - idx_low

    interpolated = normalized[idx_low] * (1.0 - frac) + normalized[idx_high] * frac

    # 映射到 [-pi/2, pi/2]
    params = np.pi * interpolated - np.pi / 2.0
    return params.reshape(n_layers, n_qubits)


class ReactionDiffusionFeatureMap:
    """
    基于反应扩散动力学的量子特征映射器。
    利用 Gray-Scott 方程生成的斑图模式作为参数化量子电路的初始参数源。
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        D_u: float = 8.0e-5,
        D_v: float = 4.0e-5,
        gamma: float = 0.024,
        kappa: float = 0.06
    ):
        if n_qubits <= 0 or n_layers <= 0:
            raise ValueError("n_qubits and n_layers must be positive")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.D_u = D_u
        self.D_v = D_v
        self.gamma = gamma
        self.kappa = kappa
        self._cached_pattern: Optional[np.ndarray] = None

    def generate_pattern(self, grid_size: int = 32, n_steps: int = 3000) -> np.ndarray:
        """生成并缓存反应扩散斑图。"""
        if grid_size < 3:
            raise ValueError("grid_size must be >= 3")
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        U, V = gray_scott_simulation(
            nx=grid_size, ny=grid_size, n_steps=n_steps,
            D_u=self.D_u, D_v=self.D_v, gamma=self.gamma, kappa=self.kappa
        )
        # 使用 U 场作为特征模式
        self._cached_pattern = U
        return U

    def get_parameters(self, data_point: np.ndarray) -> np.ndarray:
        """
        将经典数据点映射为量子电路参数。
        结合反应扩散斑图与数据点特征，生成旋转门参数。
        """
        if self._cached_pattern is None:
            self.generate_pattern()

        if len(data_point) != self.n_qubits:
            raise ValueError(
                f"Data point dimension {len(data_point)} must match n_qubits {self.n_qubits}"
            )

        base_params = pattern_to_quantum_parameters(
            self._cached_pattern, self.n_qubits, self.n_layers
        )

        # 将数据点信息编码到参数中
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                base_params[l, q] += data_point[q] * np.pi / 2.0
                # 保持在 [-pi, pi] 范围内
                base_params[l, q] = ((base_params[l, q] + np.pi) % (2.0 * np.pi)) - np.pi

        return base_params
