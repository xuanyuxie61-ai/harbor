"""
reaction_diffusion.py
=====================
神经激活模式反应-扩散模块

基于种子项目:
  - 487_gray_scott_pde: Gray-Scott 反应-扩散方程

科学背景:
  人工耳蜗刺激后，神经激活在耳蜗内呈时空分布。
  可用耦合反应-扩散方程描述:

  神经元群体激活密度 u(x,t) 和神经递质/抑制因子 v(x,t):
      ∂u/∂t = D_u ∇²u - u v² + γ (1 - u)  + I_e(x,t)
      ∂v/∂t = D_v ∇²v + u v² - (γ + κ) v

  其中:
    - D_u, D_v: 扩散系数 (代表电信号传播和神经递质扩散)
    - γ: 神经元恢复率
    - κ: 抑制因子衰减率
    - I_e(x,t): 外部电刺激输入

  该模型是 Gray-Scott 模型的生物物理变体，
  用于模拟电刺激后在螺旋神经节中的激活斑图形成。
"""

import numpy as np


def neural_activation_rd(U, V, dx, dy, dt, D_u, D_v, gamma, kappa,
                          stimulus, laplacian_func=None):
    """
    单步 Euler 更新神经激活反应-扩散方程。

    Parameters
    ----------
    U, V : ndarray, shape (nx, ny)
        当前激活密度和抑制因子
    dx, dy : float
        空间步长
    dt : float
        时间步长
    D_u, D_v : float
        扩散系数
    gamma, kappa : float
        反应速率参数
    stimulus : ndarray, shape (nx, ny)
        外部刺激输入
    laplacian_func : callable or None
        自定义 Laplacian 函数。若为 None 使用 5点 stencil。

    Returns
    -------
    U_new, V_new : ndarray
        更新后的场
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    stimulus = np.asarray(stimulus, dtype=float)

    if U.shape != V.shape or U.shape != stimulus.shape:
        raise ValueError("U, V, stimulus 形状必须相同")

    if laplacian_func is None:
        from laplacian_operator import laplacian_5point
        laplacian_func = laplacian_5point

    Lu = laplacian_func(U, dx, dy)
    Lv = laplacian_func(V, dx, dy)

    # 反应项
    reaction_u = -U * V**2 + gamma * (1.0 - U) + stimulus
    reaction_v = U * V**2 - (gamma + kappa) * V

    U_new = U + dt * (D_u * Lu + reaction_u)
    V_new = V + dt * (D_v * Lv + reaction_v)

    # 边界处理与截断
    U_new = np.clip(U_new, 0.0, 1.0)
    V_new = np.clip(V_new, 0.0, 1.0)

    return U_new, V_new


class NeuralActivationPattern:
    """
    神经激活时空模式求解器。
    """

    def __init__(self, nx, ny, dx, dy, D_u=0.01, D_v=0.005,
                 gamma=0.024, kappa=0.06):
        """
        Parameters
        ----------
        nx, ny : int
            网格数
        dx, dy : float
            空间步长 (mm)
        D_u, D_v : float
            扩散系数 (mm²/ms)
        gamma, kappa : float
            反应参数
        """
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.D_u = float(D_u)
        self.D_v = float(D_v)
        self.gamma = float(gamma)
        self.kappa = float(kappa)

        # CFL 条件检查
        dt_max = 0.25 * min(dx**2, dy**2) / max(D_u, D_v)
        self.dt_max = dt_max

        self.U = None
        self.V = None
        self._initialized = False

    def initialize(self, seed_pattern='gaussian'):
        """
        初始化激活场。

        Parameters
        ----------
        seed_pattern : str
            'gaussian', 'random', 'uniform'
        """
        if seed_pattern == 'gaussian':
            cx, cy = self.nx // 2, self.ny // 2
            X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
            sigma = min(self.nx, self.ny) / 8.0
            V_seed = 0.25 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
            U = np.ones((self.nx, self.ny)) - 2.0 * V_seed
        elif seed_pattern == 'random':
            np.random.seed(42)
            U = np.random.rand(self.nx, self.ny) * 0.1 + 0.9
            V = np.random.rand(self.nx, self.ny) * 0.1
            self.U = np.clip(U, 0.0, 1.0)
            self.V = np.clip(V, 0.0, 1.0)
            self._initialized = True
            return
        elif seed_pattern == 'uniform':
            U = np.ones((self.nx, self.ny))
            V = np.zeros((self.nx, self.ny))
            self.U = U
            self.V = V
            self._initialized = True
            return
        else:
            raise ValueError(f"未知的 seed_pattern: {seed_pattern}")

        self.U = np.clip(U, 0.0, 1.0)
        self.V = np.clip(V_seed, 0.0, 1.0)
        self._initialized = True

    def evolve(self, n_steps, stimulus_history=None, dt=None):
        """
        时间演化。

        Parameters
        ----------
        n_steps : int
            时间步数
        stimulus_history : list of ndarray or None
            每步的刺激场
        dt : float or None
            时间步长，若为 None 则使用 dt_max

        Returns
        -------
        U_history : list
            激活密度历史
        V_history : list
            抑制因子历史
        """
        if not self._initialized:
            raise RuntimeError("必须先调用 initialize()")

        if dt is None:
            dt = self.dt_max * 0.5
        if dt > self.dt_max:
            raise ValueError(f"dt={dt} 超过 CFL 限制 {self.dt_max}")

        U_history = []
        V_history = []

        for step in range(n_steps):
            if stimulus_history is not None and step < len(stimulus_history):
                stim = stimulus_history[step]
            else:
                stim = np.zeros((self.nx, self.ny))

            self.U, self.V = neural_activation_rd(
                self.U, self.V, self.dx, self.dy, dt,
                self.D_u, self.D_v, self.gamma, self.kappa, stim
            )
            U_history.append(self.U.copy())
            V_history.append(self.V.copy())

        return U_history, V_history

    def compute_spread_metrics(self):
        """
        计算激活区域的空间展宽指标。

        Returns
        -------
        active_area : float
            激活区域面积 (像素数)
        centroid : tuple
            质心坐标
        spread_std : float
            标准差展宽
        """
        if self.U is None:
            raise RuntimeError("未初始化")

        threshold = 0.5
        active_mask = self.U > threshold
        active_area = np.sum(active_mask)

        if active_area < 1:
            return 0.0, (0.0, 0.0), 0.0

        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        centroid_x = np.sum(X * active_mask) / active_area
        centroid_y = np.sum(Y * active_mask) / active_area

        spread_std = np.sqrt(
            np.sum(((X - centroid_x)**2 + (Y - centroid_y)**2) * active_mask)
            / active_area
        )

        return float(active_area), (float(centroid_x), float(centroid_y)), float(spread_std)
