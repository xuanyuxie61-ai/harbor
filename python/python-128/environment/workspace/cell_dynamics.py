"""
cell_dynamics.py
================
三维椭球形细胞迁移动力学与敏感性分析

融合原始项目：
  - 332_ellipsoid：椭球表面积（不完全椭圆积分）
  - 1064_sensitive_ode：初值敏感 ODE 系统
  - 058_atkinson：数值 ODE 与求根方法

数学物理模型：
  1. 椭球细胞几何模型：
       (x/a)² + (y/b)² + (z/c)² = 1
     表面积公式（Knud Thomsen 近似）：
       S ≈ 4π [ (a^p b^p + a^p c^p + b^p c^p) / 3 ]^{1/p}
       其中 p ≈ 1.6075（Rudolf 近似）。
     更精确地，通过不完全椭圆积分：
       S = 2π c² + 2πab / sin(φ) · [ E(φ,m) sin²φ + F(φ,m) cos²φ ]
       φ = arccos(c/a),  m = a²(b²-c²) / [b²(a²-c²)]

  2. 细胞迁移 ODE（扩展 Keller-Segel 模型到个体细胞）：
       dX/dt = μ · ∇c / (1 + γ |∇c|)   (chemotaxis 速度)
             + σ · ξ(t)                  (随机游动)
             - β · v_ECM(X)              (ECM 阻力)
     其中 ξ(t) 为白噪声，μ 为趋化敏感系数，γ 为饱和参数。

  3. 敏感性方程（源自 sensitive_ode 思想）：
       设初始条件有扰动 ε，则敏感变量 s(t) = ∂X/∂ε 满足：
       ds/dt = J_f(X(t)) · s(t)
       其中 J_f 为速度场关于位置的 Jacobian。
"""

import numpy as np
from special_math import jacobi_elliptic


# ---------------------------------------------------------------------------
# Ellipsoid Geometry (from 332_ellipsoid)
# ---------------------------------------------------------------------------
def ellipsoid_surface_area_rudolf(a: float, b: float, c: float, p: float = 1.6075):
    """
    Knud Thomsen / Rudolf 近似公式计算椭球表面积。

    公式：
        S ≈ 4π [ ( (ab)^p + (ac)^p + (bc)^p ) / 3 ]^{1/p}

    参数
    ----
    a, b, c : float
        三个半轴长度（将自动按大小排序）
    p : float
        形状参数，默认 1.6075

    返回
    ----
    area : float
    """
    a, b, c = abs(float(a)), abs(float(b)), abs(float(c))
    # 排序确保 a ≥ b ≥ c
    arr = sorted([a, b, c], reverse=True)
    a, b, c = arr[0], arr[1], arr[2]
    if a < 1e-15 or b < 1e-15 or c < 1e-15:
        raise ValueError("ellipsoid_surface_area_rudolf: 半轴必须为正")
    term = ((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0
    return 4.0 * np.pi * (term ** (1.0 / p))


def ellipsoid_surface_area_elliptic(a: float, b: float, c: float):
    """
    使用不完全椭圆积分计算椭球表面积（源自 332_ellipsoid 核心思想）。

    要求 a ≥ b ≥ c > 0。令：
        φ = arccos(c / a)
        m = a²(b² - c²) / [ b²(a² - c²) ]   (模数)
    则表面积：
        S = 2π c² + 2π a b · temp2
    其中 temp = E(φ,m) sin²φ + F(φ,m) cos²φ，
          temp2 = temp / sin φ。

    这里使用 Jacobi 椭圆函数与第二类椭圆积分的关系进行数值计算。
    """
    a, b, c = abs(float(a)), abs(float(b)), abs(float(c))
    arr = sorted([a, b, c], reverse=True)
    a, b, c = arr[0], arr[1], arr[2]
    if a < 1e-15 or c < 1e-15:
        raise ValueError("ellipsoid_surface_area_elliptic: 半轴必须为正")

    phi = np.arccos(np.clip(c / a, -1.0, 1.0))
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    denom = b * b * (a * a - c * c)
    if abs(denom) < 1e-15:
        m = 1.0
    else:
        m = (a * a * (b * b - c * c)) / denom
        m = np.clip(m, 0.0, 1.0)

    # 通过 Jacobi 椭圆函数关联计算第二类不完全椭圆积分 E(φ,m)
    # E(φ,m) = ∫_0^φ √(1 - m sin²θ) dθ
    # 使用数值积分（Simpson 规则）
    def integrand_E(theta):
        return np.sqrt(1.0 - m * np.sin(theta) ** 2)

    def integrand_F(theta):
        return 1.0 / np.sqrt(1.0 - m * np.sin(theta) ** 2)

    n_quad = 200
    theta = np.linspace(0.0, phi, n_quad)
    dth = phi / (n_quad - 1)
    E_val = np.trapz(integrand_E(theta), theta)
    F_val = np.trapz(integrand_F(theta), theta)

    temp = E_val * sin_phi ** 2 + F_val * cos_phi ** 2
    if abs(sin_phi) < 1e-15:
        temp2 = 1.0
    else:
        temp2 = temp / sin_phi

    return 2.0 * np.pi * (c ** 2 + a * b * temp2)


def ellipsoid_volume(a: float, b: float, c: float):
    """
    椭球体积：V = (4/3) π a b c
    """
    a, b, c = abs(float(a)), abs(float(b)), abs(float(c))
    return (4.0 / 3.0) * np.pi * a * b * c


# ---------------------------------------------------------------------------
# Cell Migration ODE System (from 1064_sensitive_ode + Keller-Segel)
# ---------------------------------------------------------------------------
class CellAgent:
    """
    单个椭球形细胞迁移代理。

    状态变量：
        position : np.ndarray, shape (3,)
        velocity : np.ndarray, shape (3,)
        shape    : tuple (a, b, c)  三个半轴
        phase    : int  细胞周期相位 (0=G1, 1=S, 2=G2, 3=M)
    """

    def __init__(self, position, shape=(5.0, 3.0, 2.0), phase=0):
        self.position = np.asarray(position, dtype=float).reshape(3)
        self.shape = tuple(float(s) for s in shape)
        self.phase = int(phase) % 4
        self.velocity = np.zeros(3, dtype=float)
        self.sensitivity = 1.0

    def chemotaxis_velocity(self, grad_c, mu=0.5, gamma=0.3):
        """
        计算 chemotaxis 迁移速度（饱和模型）：

            v_chemo = μ · ∇c / (1 + γ |∇c|)

        参数
        ----
        grad_c : np.ndarray, shape (3,)
            趋化因子浓度梯度
        mu : float
            最大趋化速度
        gamma : float
            饱和参数

        返回
        ----
        v : np.ndarray, shape (3,)

        TODO (Hole 3): 实现 Keller-Segel 饱和 chemotaxis 速度公式。
        需要根据梯度范数计算饱和因子，并处理零梯度退化情形。
        """
        # === HOLE 3 BEGIN ===
        raise NotImplementedError("Hole 3: chemotaxis_velocity 尚未实现")
        # === HOLE 3 END ===

    def ecm_drag(self, ecm_density_func, beta=0.2):
        """
        计算 ECM 阻力（与局部 ECM 密度成正比）。

        公式：
            v_drag = -β · ρ_ECM(x) · v
        这里简化为仅返回阻力系数。
        """
        rho = ecm_density_func(self.position)
        return -beta * rho

    def stochastic_component(self, sigma=0.05):
        """
        随机游动分量（ persistent random walk ）。

        公式：
            ξ ~ N(0, σ² I)
        """
        return np.random.normal(0.0, sigma, size=3)

    def step(self, grad_c, dt, ecm_density_func=None, mu=0.5, gamma=0.3,
             beta=0.2, sigma=0.05):
        """
        推进细胞位置一个时间步。

        综合速度场：
            v = v_chemo + v_stochastic + v_drag
            x_{n+1} = x_n + dt · v
        """
        v = self.chemotaxis_velocity(grad_c, mu, gamma)
        v += self.stochastic_component(sigma)
        if ecm_density_func is not None:
            drag = self.ecm_drag(ecm_density_func, beta)
            # 阻力作用在当前速度上
            v *= np.exp(drag * dt)
        self.velocity = v
        self.position += dt * v
        return self.position


class CellPopulation:
    """
    细胞群体，管理多个 CellAgent，并执行敏感性分析。
    """

    def __init__(self, n_cells: int = 50, domain=((-1, 1), (-1, 1), (-0.5, 0.5))):
        self.n_cells = int(n_cells)
        self.domain = domain
        self.cells = []
        for _ in range(self.n_cells):
            pos = np.array([
                np.random.uniform(domain[0][0], domain[0][1]),
                np.random.uniform(domain[1][0], domain[1][1]),
                np.random.uniform(domain[2][0], domain[2][1]),
            ])
            # 随机椭球形状
            shape = tuple(sorted([
                np.random.uniform(3.0, 7.0),
                np.random.uniform(2.0, 5.0),
                np.random.uniform(1.5, 4.0),
            ], reverse=True))
            self.cells.append(CellAgent(pos, shape))

    def compute_mean_position(self):
        """计算群体平均位置。"""
        if not self.cells:
            return np.zeros(3)
        return np.mean([c.position for c in self.cells], axis=0)

    def compute_spread(self):
        """计算群体位置的标准差（扩散度量）。"""
        if len(self.cells) < 2:
            return 0.0
        pos = np.array([c.position for c in self.cells])
        return np.mean(np.std(pos, axis=0))

    def sensitivity_analysis(self, grad_c_func, dt, n_steps=10, eps=1e-4):
        """
        执行初始条件敏感性分析（源自 sensitive_ode 思想）。

        对单个代表性细胞，比较初始条件扰动 ε 后的轨迹差异：
            δ(t) = |X(t; X_0+ε) - X(t; X_0)|

        返回随时间变化的 L2 偏差序列。
        """
        if not self.cells:
            return np.array([])
        base_cell = self.cells[0]
        traj_base = []
        traj_pert = []
        pos0 = base_cell.position.copy()
        pos_pert = pos0 + eps * np.ones(3)

        cell_base = CellAgent(pos0, base_cell.shape)
        cell_pert = CellAgent(pos_pert, base_cell.shape)

        for _ in range(n_steps):
            g = grad_c_func(cell_base.position)
            cell_base.step(g, dt)
            traj_base.append(cell_base.position.copy())

            g = grad_c_func(cell_pert.position)
            cell_pert.step(g, dt)
            traj_pert.append(cell_pert.position.copy())

        diff = np.array([
            np.linalg.norm(traj_base[i] - traj_pert[i])
            for i in range(n_steps)
        ])
        return diff

    def step_all(self, grad_c_func, dt, **kwargs):
        """
        对所有细胞推进一个时间步。

        参数
        ----
        grad_c_func : callable
            grad_c_func(position) -> gradient_vector
        """
        for cell in self.cells:
            g = grad_c_func(cell.position)
            cell.step(g, dt, **kwargs)

    def total_surface_area(self):
        """计算所有细胞的表面积之和。"""
        total = 0.0
        for cell in self.cells:
            total += ellipsoid_surface_area_rudolf(*cell.shape)
        return total

    def total_volume(self):
        """计算所有细胞的体积之和。"""
        total = 0.0
        for cell in self.cells:
            total += ellipsoid_volume(*cell.shape)
        return total
