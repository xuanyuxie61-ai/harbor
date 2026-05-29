"""
parameter_inversion.py
======================
基于 paraheat_functional (846_paraheat_functional) 的参数化热方程求解与反演框架，
用于估计海气耦合模式中的关键热力学参数（热扩散系数、Newton 冷却系数、
风应力-热通量耦合强度）。

科学背景
--------
海洋热含量变化由热力学方程控制：

∂T/∂t = D_h * ∇²T - (1/τ) * T + Q(x,y,t) / (ρ_0 * c_p * H)

其中：
- D_h : 水平热扩散系数（m²/s），控制热量在海洋中的传播速度；
- τ   : Newton 冷却时间尺度（s），表征海表向大气的热损失；
- Q   : 海表热通量（W/m²），由风应力与 SST 异常耦合决定。

参数反演问题：给定观测 SST 场 {T_obs(x_i, y_i, t_j)}，
寻找最优参数 θ = (D_h, τ, coupling_strength) 使得模型输出与观测的残差最小：

J(θ) = (1/2) * Σ_{i,j} (T_model(x_i,y_i,t_j; θ) - T_obs(x_i,y_i,t_j))²

本模块实现：
1. 二维稳态热方程的有限元/有限差分求解；
2. 热扩散系数的分段常数参数化；
3. 梯度下降法参数反演。

核心公式
--------
1. 稳态热方程（忽略时间导数）：
   
   -D_h * ∇²T + (1/τ) * T = Q / (ρ_0 * c_p * H)

2. 二维五点差分离散：
   -D_h * [(T_{i-1,j} - 2T_{i,j} + T_{i+1,j})/dx²
         + (T_{i,j-1} - 2T_{i,j} + T_{i,j+1})/dy²]
   + (1/τ) * T_{i,j} = Q_{i,j} / (ρ_0 * c_p * H)

3. 目标泛函（最小二乘）：
   
   J(θ) = (1/2) * ||T_model(θ) - T_obs||²_2 + (λ/2) * ||θ - θ_prior||²_2

4. 梯度（灵敏度方法）：
   
   ∂J/∂θ_k = Σ_{i,j} (T_model - T_obs)_{i,j} * (∂T_model/∂θ_k)_{i,j}
            + λ * (θ_k - θ_prior_k)

5. 灵敏度方程（以 D_h 为例）：
   
   -D_h * ∇²(∂T/∂D_h) + (1/τ) * (∂T/∂D_h) = ∇²T
"""

import numpy as np
from typing import Tuple, Callable, Optional


def solve_steady_heat_2d(nx: int, ny: int, dx: float, dy: float,
                         diffusivity: float, tau: float,
                         heat_source: np.ndarray,
                         boundary_value: float = 0.0) -> np.ndarray:
    """
    求解二维稳态热方程的有限差分解。

    方程：
    -D * ∇²T + (1/τ) * T = Q

    参数
    ----
    nx, ny : int
        网格点数。
    dx, dy : float
        网格间距。
    diffusivity : float
        热扩散系数 D_h。
    tau : float
        Newton 冷却时间尺度。
    heat_source : np.ndarray, shape (nx, ny)
        热源项 Q。
    boundary_value : float
        Dirichlet 边界值。

    返回
    ----
    T : np.ndarray, shape (nx, ny)
        温度场。
    """
    if heat_source.shape != (nx, ny):
        raise ValueError("heat_source shape mismatch")
    if diffusivity <= 0 or tau <= 0:
        raise ValueError("diffusivity and tau must be positive")

    n = nx * ny
    # 构建稀疏矩阵（用稠密矩阵简化，小规模问题）
    A = np.zeros((n, n), dtype=float)
    b = heat_source.ravel().copy()

    cx = diffusivity / (dx * dx)
    cy = diffusivity / (dy * dy)
    c_inv_tau = 1.0 / tau

    for j in range(ny):
        for i in range(nx):
            row = j * nx + i
            diag = 0.0

            # 左
            if i > 0:
                A[row, row - 1] = -cx
                diag += cx
            else:
                b[row] += cx * boundary_value
                diag += cx

            # 右
            if i < nx - 1:
                A[row, row + 1] = -cx
                diag += cx
            else:
                b[row] += cx * boundary_value
                diag += cx

            # 下
            if j > 0:
                A[row, row - nx] = -cy
                diag += cy
            else:
                b[row] += cy * boundary_value
                diag += cy

            # 上
            if j < ny - 1:
                A[row, row + nx] = -cy
                diag += cy
            else:
                b[row] += cy * boundary_value
                diag += cy

            # 对角元：Newton 冷却 + 扩散中心
            A[row, row] = diag + c_inv_tau

    # 求解线性系统
    T = np.linalg.solve(A, b)
    return T.reshape((nx, ny))


def piecewise_diffusivity(nx: int, ny: int,
                          param_blocks: np.ndarray,
                          x_breaks: np.ndarray,
                          y_breaks: np.ndarray) -> np.ndarray:
    """
    构造分段常数热扩散系数场。

    参数
    ----
    nx, ny : int
        网格维度。
    param_blocks : np.ndarray, shape (nxc, nyc)
        各子区域的扩散系数值。
    x_breaks : np.ndarray, shape (nxc+1,)
        x 方向断点。
    y_breaks : np.ndarray, shape (nyc+1,)
        y 方向断点。

    返回
    ----
    D : np.ndarray, shape (nx, ny)
        扩散系数场。
    """
    nxc, nyc = param_blocks.shape
    if x_breaks.shape[0] != nxc + 1 or y_breaks.shape[0] != nyc + 1:
        raise ValueError("Breaks array dimension mismatch")

    D = np.zeros((nx, ny), dtype=float)
    x_grid = np.linspace(x_breaks[0], x_breaks[-1], nx)
    y_grid = np.linspace(y_breaks[0], y_breaks[-1], ny)

    for j in range(ny):
        for i in range(nx):
            x, y = x_grid[i], y_grid[j]
            # 找到所属子块
            ix = min(nxc - 1, max(0, np.searchsorted(x_breaks[1:], x)))
            iy = min(nyc - 1, max(0, np.searchsorted(y_breaks[1:], y)))
            D[i, j] = param_blocks[ix, iy]

    return D


def objective_function(theta: np.ndarray,
                       T_obs: np.ndarray,
                       nx: int, ny: int, dx: float, dy: float,
                       heat_source: np.ndarray,
                       theta_prior: np.ndarray,
                       lam: float = 0.01) -> float:
    """
    计算参数反演的目标泛函 J(θ)。

    参数
    ----
    theta : np.ndarray, shape (3,)
        参数向量 [D_h, τ, coupling_strength]。
    T_obs : np.ndarray, shape (nx, ny)
        观测温度场。
    lam : float
        正则化系数。
    theta_prior : np.ndarray
        先验参数值。

    返回
    ----
    J : float
        目标函数值。
    """
    D_h, tau, coupling = theta[0], theta[1], theta[2]

    # 边界处理
    D_h = max(D_h, 1e-6)
    tau = max(tau, 1e-6)
    coupling = np.clip(coupling, -10.0, 10.0)

    # 热源包含耦合项
    Q_eff = heat_source + coupling * T_obs

    T_model = solve_steady_heat_2d(nx, ny, dx, dy, D_h, tau, Q_eff)

    residual = T_model - T_obs
    data_misfit = 0.5 * np.sum(residual ** 2)
    regularization = 0.5 * lam * np.sum((theta - theta_prior) ** 2)

    return data_misfit + regularization


def gradient_descent_inversion(T_obs: np.ndarray,
                               nx: int, ny: int, dx: float, dy: float,
                               heat_source: np.ndarray,
                               theta_init: np.ndarray,
                               theta_prior: np.ndarray,
                               lr: float = 0.01,
                               n_iter: int = 100,
                               lam: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用梯度下降法反演热力学参数。

    参数归一化策略：
    - D_h 的量级约为 10^3，τ 的量级约为 10^6，coupling 的量级约为 10^{-2}。
    - 直接梯度下降会因量纲差异而失效。
    - 本实现采用归一化梯度：对每个参数分别估计其影响尺度，
      然后使用归一化梯度进行下降。
    """
    theta = theta_init.copy().astype(float)
    history = np.zeros(n_iter)

    # 各参数的有限差分步长（匹配量纲）
    eps_vals = np.array([10.0, 1.0 * 24 * 3600, 0.001])
    # 各参数的尺度因子（用于梯度归一化）
    scales = np.array([1000.0, 20.0 * 24 * 3600, 0.01])

    for it in range(n_iter):
        J0 = objective_function(theta, T_obs, nx, ny, dx, dy,
                                heat_source, theta_prior, lam)
        history[it] = J0

        if not np.isfinite(J0):
            break

        grad = np.zeros(3)
        for k in range(3):
            theta_plus = theta.copy()
            theta_plus[k] += eps_vals[k]
            J_plus = objective_function(theta_plus, T_obs, nx, ny, dx, dy,
                                        heat_source, theta_prior, lam)
            if np.isfinite(J_plus):
                grad[k] = (J_plus - J0) / eps_vals[k]
            else:
                grad[k] = 0.0

        # 归一化梯度
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-14:
            grad = grad / grad_norm

        # 按参数尺度分别更新
        step = lr * grad * scales
        theta = theta - step

        # 参数边界与数值鲁棒性
        theta[0] = max(theta[0], 100.0)
        theta[1] = max(theta[1], 1.0 * 24 * 3600)
        theta[2] = np.clip(theta[2], -1.0, 1.0)

        # 自适应学习率
        if it > 0 and history[it] > history[it - 1]:
            lr *= 0.5
            if lr < 1e-6:
                break

    return theta, history
    """
    使用梯度下降法反演热力学参数。

    参数
    ----
    T_obs : np.ndarray
        观测温度场。
    theta_init : np.ndarray
        初始参数猜测。
    theta_prior : np.ndarray
        先验参数。
    lr : float
        学习率。
    n_iter : int
        迭代次数。

    返回
    ----
    theta_opt : np.ndarray
        优化后的参数。
    history : np.ndarray
        目标函数历史。
    """
    theta = theta_init.copy()
    history = np.zeros(n_iter)

    # 有限差分梯度
    eps = 1e-5

    for it in range(n_iter):
        J0 = objective_function(theta, T_obs, nx, ny, dx, dy,
                                heat_source, theta_prior, lam)
        history[it] = J0

        grad = np.zeros(3)
        for k in range(3):
            theta_plus = theta.copy()
            theta_plus[k] += eps
            J_plus = objective_function(theta_plus, T_obs, nx, ny, dx, dy,
                                        heat_source, theta_prior, lam)
            grad[k] = (J_plus - J0) / eps

        # 梯度下降步
        theta = theta - lr * grad

        # 参数边界
        theta[0] = max(theta[0], 1e-6)
        theta[1] = max(theta[1], 1e-6)
        theta[2] = np.clip(theta[2], -10.0, 10.0)

        # 自适应学习率
        if it > 0 and history[it] > history[it - 1]:
            lr *= 0.5

    return theta, history


def sensitivity_analysis(theta: np.ndarray,
                         T_obs: np.ndarray,
                         nx: int, ny: int, dx: float, dy: float,
                         heat_source: np.ndarray,
                         perturbation: float = 0.1) -> dict:
    """
    参数灵敏度分析。

    计算每个参数扰动 ±10% 时模型输出的相对变化。

    返回
    ----
    sensitivities : dict
        各参数的灵敏度指标。
    """
    base_model = solve_steady_heat_2d(nx, ny, dx, dy,
                                      max(theta[0], 1e-6),
                                      max(theta[1], 1e-6),
                                      heat_source + theta[2] * T_obs)

    sens = {}
    param_names = ["diffusivity", "tau", "coupling"]
    for k in range(3):
        theta_plus = theta.copy()
        theta_plus[k] *= (1.0 + perturbation)
        model_plus = solve_steady_heat_2d(nx, ny, dx, dy,
                                          max(theta_plus[0], 1e-6),
                                          max(theta_plus[1], 1e-6),
                                          heat_source + theta_plus[2] * T_obs)
        rel_change = np.linalg.norm(model_plus - base_model) / (np.linalg.norm(base_model) + 1e-14)
        sens[param_names[k]] = float(rel_change / perturbation)

    return sens
