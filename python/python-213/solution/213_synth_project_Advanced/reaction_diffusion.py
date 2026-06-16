"""
reaction_diffusion.py - 反应-扩散动力学模块
============================================

融合种子项目:
  - 486_gray_scott_movie: Gray-Scott 反应-扩散系统
  - 433_fisher_exact: Fisher-KPP 方程精确解

核心数学: 反应-扩散 PDE 及其在凸优化中的应用

反应-扩散方程是凸优化中 PDE 约束的核心.

Gray-Scott 模型 (自催化反应):
  du/dt = D_u * Laplacian(u) - u*v^2 + f*(1-u)
  dv/dt = D_v * Laplacian(v) + u*v^2 - (f+k)*v

  物理意义:
    u: 反应物 U 的浓度
    v: 产物 V 的浓度
    f: 流速 (feed rate)
    k: 灭活率 (kill rate)
    D_u, D_v: 扩散系数

  反应动力学:
    R_u(u,v) = -u*v^2 + f*(1-u)
    R_v(u,v) = +u*v^2 - (f+k)*v

  均匀稳态:
    (u_0, v_0) = (1, 0)  或  (u_*, v_*) 其中
    u_* = (f+k) / (2*k),  v_* 由质量守恒确定.

Fisher-KPP 方程:
  du/dt = D * u_xx + r * u * (1 - u)

  行波解 (Ablowitz & Zeppetella, 1979):
    u(x,t) = 1 / (1 + a * exp(k*(x - c*t)))^2
  其中:
    c = 5/sqrt(6) * sqrt(D*r)  (最小波速)
    k = -1/sqrt(6) * sqrt(r/D)
    a = 任意正常数 (波位置参数)

  Fisher 方程的凸性:
    反应项 f(u) = r*u*(1-u) 在 [0,1] 上是凹的.
    这保证了 PDE 约束优化中的凸性.

在最优控制中的应用:
  将控制变量 q(x) 引入反应项:
    du/dt = D * Laplacian(u) + R(u) + q(x)*u

  优化问题:
    min_{q} integral (u(T) - u_target)^2 dx + gamma * |q|^2 dx
    s.t. PDE 约束
         q_min <= q <= q_max

  有限元离散化后:
    M dU/dt + K U = F(U) + G*Q
    => 线性等式约束 (给定 U^n 求 U^{n+1})
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Dict, Optional


class GrayScottModel:
    """
    Gray-Scott 反应-扩散模型 (融合 486_gray_scott_movie).

    反应项:
      R_u = -u*v^2 + f*(1-u)
      R_v = +u*v^2 - (f+k)*v

    Jacobian (用于 Newton 线性化):
      dR_u/du = -v^2 - f
      dR_u/dv = -2*u*v
      dR_v/du = +v^2
      dR_v/dv = +2*u*v - (f+k)

    在最优控制框架中, Gray-Scott 模型提供:
      1. 目标图案的生成 (self-organization)
      2. 非线性反应项带来的优化挑战
      3. 多稳态特性导致非凸优化

    Parameters
    ----------
    f : float
        流速 (feed rate), 典型范围 [0.01, 0.08]
    k : float
        灭活率 (kill rate), 典型范围 [0.04, 0.07]
    D_u : float
        U 的扩散系数
    D_v : float
        V 的扩散系数
    """

    def __init__(self, f: float = 0.04, k: float = 0.06,
                 D_u: float = 0.16, D_v: float = 0.08):
        self.f = f
        self.k = k
        self.D_u = D_u
        self.D_v = D_v

    def reaction(self, u: np.ndarray, v: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算反应项.

        R_u(u,v) = -u*v^2 + f*(1-u)
        R_v(u,v) = +u*v^2 - (f+k)*v

        Parameters
        ----------
        u, v : ndarray
            浓度场

        Returns
        -------
        R_u, R_v : ndarray
            反应率
        """
        R_u = -u * v**2 + self.f * (1.0 - u)
        R_v = u * v**2 - (self.f + self.k) * v
        return R_u, R_v

    def reaction_jacobian(self, u: np.ndarray, v: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
        """
        计算反应项的 Jacobian 矩阵 (逐点).

        J = [ dR_u/du  dR_u/dv ] = [ -v^2-f    -2uv ]
            [ dR_v/du  dR_v/dv ]   [  v^2      2uv-(f+k) ]

        用于反应项的隐式处理或 Newton 线性化.

        Returns
        -------
        J_uu, J_uv, J_vu, J_vv : ndarray
            Jacobian 分量
        """
        J_uu = -v**2 - self.f
        J_uv = -2.0 * u * v
        J_vu = v**2
        J_vv = 2.0 * u * v - (self.f + self.k)
        return J_uu, J_uv, J_vu, J_vv

    def uniform_steady_state(self) -> Tuple[float, float, float, float]:
        """
        计算均匀稳态.

        稳态条件: R_u = R_v = 0

        平庸稳态: (u, v) = (1, 0)
        非平庸稳态:
          u_* + v_* = 1  (从 R_u + R_v = 0)
          u_* * v_*^2 = (f+k) * v_*  =>  u_* * v_* = f+k
          => (1-v_*) * v_* = f+k
          => v_*^2 - v_* + (f+k) = 0
          => v_* = (1 - sqrt(1 - 4(f+k))) / 2

        Returns
        -------
        u_trivial, v_trivial, u_nontrivial, v_nontrivial : float
            稳态值
        """
        u_tri, v_tri = 1.0, 0.0

        disc = 1.0 - 4.0 * (self.f + self.k)
        if disc >= 0:
            v_nt = (1.0 - np.sqrt(disc)) / 2.0
            u_nt = 1.0 - v_nt
        else:
            u_nt, v_nt = np.nan, np.nan

        return u_tri, v_tri, u_nt, v_nt

    def evolve_step(self, u: np.ndarray, v: np.ndarray,
                     laplacian_u: np.ndarray, laplacian_v: np.ndarray,
                     dt: float, control: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        一步向前 Euler 时间推进.

        u^{n+1} = u^n + dt * (D_u * Lap(u^n) + R_u(u^n, v^n) + q*u^n)
        v^{n+1} = v^n + dt * (D_v * Lap(v^n) + R_v(u^n, v^n))

        Parameters
        ----------
        u, v : ndarray
            当前浓度场
        laplacian_u, laplacian_v : ndarray
            Laplacian 近似
        dt : float
            时间步长
        control : ndarray, optional
            控制变量 q(x)

        Returns
        -------
        u_new, v_new : ndarray
            更新后的浓度场
        """
        R_u, R_v = self.reaction(u, v)

        du = self.D_u * laplacian_u + R_u
        dv = self.D_v * laplacian_v + R_v

        if control is not None:
            du += control * u

        u_new = u + dt * du
        v_new = v + dt * dv

        # 物理约束: 浓度非负
        u_new = np.maximum(u_new, 0.0)
        v_new = np.maximum(v_new, 0.0)

        return u_new, v_new

    def compute_laplacian_9pt(self, field: np.ndarray,
                               dx: float, dy: float) -> np.ndarray:
        """
        9 点 Laplacian 近似 (融合 486_gray_scott_movie).

        在二维矩形网格上, 使用 9 点差分模板:

        Lap u_{i,j} = (1/6h^2) * [
            u_{i-1,j-1} + u_{i+1,j-1} + u_{i-1,j+1} + u_{i+1,j+1}
            + 4*(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1})
            - 20*u_{i,j}
        ]

        这是 O(h^4) 精度的各向同性近似 (对 torus 周期边界).

        Parameters
        ----------
        field : ndarray, shape (nx, ny)
            场值
        dx, dy : float
            网格间距

        Returns
        -------
        ndarray
            Laplacian 近似
        """
        nx, ny = field.shape
        lap = np.zeros_like(field)

        # 周期边界 (torus)
        h2 = 6.0 * dx * dy  # 假设 dx = dy

        for i in range(nx):
            for j in range(ny):
                ip = (i + 1) % nx
                im = (i - 1) % nx
                jp = (j + 1) % ny
                jm = (j - 1) % ny

                lap[i, j] = (
                    field[im, jm] + field[ip, jm] +
                    field[im, jp] + field[ip, jp] +
                    4.0 * (field[im, j] + field[ip, j] +
                           field[i, jm] + field[i, jp]) -
                    20.0 * field[i, j]
                ) / h2

        return lap


class FisherKPPEquation:
    """
    Fisher-KPP 方程 (融合 433_fisher_exact).

    du/dt = D * u_xx + r * u * (1 - u)

    行波精确解:
      u(x,t) = 1 / (1 + a*exp(k*(x-ct)))^2
      c = 5/sqrt(6) * sqrt(D*r)
      k_wave = -1/sqrt(6) * sqrt(r/D)

    在优化中用于提供:
      1. 行波目标图案
      2. 边界条件的解析表达式
      3. 收敛性验证的基准解
    """

    def __init__(self, D: float = 1.0, r: float = 1.0):
        self.D = D
        self.r = r
        self._compute_parameters()

    def _compute_parameters(self):
        """计算行波参数."""
        Dr = self.D * self.r
        self.wave_speed = 5.0 / np.sqrt(6.0) * np.sqrt(Dr)
        self.wave_number = -1.0 / np.sqrt(6.0) * np.sqrt(self.r / self.D)
        self.amplitude = 1.0  # 自由参数 a

    def exact_solution(self, t: float, x: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 Fisher 方程的精确行波解 (融合 433_fisher_exact).

        u(x,t) = 1 / (1 + a*exp(k*(x-ct)))^2
        u_t = 2c*a*k*exp(k*z) / (1 + a*exp(k*z))^3
        u_x = -2a*k*exp(k*z) / (1 + a*exp(k*z))^3

        其中 z = x - c*t.

        Parameters
        ----------
        t : float
            时间
        x : ndarray
            空间坐标

        Returns
        -------
        u, ut, ux : ndarray
            解及其时间/空间导数
        """
        a = self.amplitude
        c = self.wave_speed
        k = self.wave_number

        z = x - c * t
        exp_kz = np.exp(k * z)
        denom = 1.0 + a * exp_kz

        u = 1.0 / denom**2
        ut = 2.0 * c * a * k * exp_kz / denom**3
        ux = -2.0 * a * k * exp_kz / denom**3

        return u, ut, ux

    def convex_reaction_term(self, u: np.ndarray) -> Tuple[np.ndarray,
                                                              np.ndarray]:
        """
        计算反应项及其导数 (凸性分析).

        f(u) = r * u * (1 - u)
        f'(u) = r * (1 - 2u)
        f''(u) = -2r < 0  =>  f 是凹的

        凹性对优化意味着:
          - 反应项是凹的 => 约束集是凸的 (在适当的变换下)
          - 保证了全局最优性 (在凸框架中)

        Parameters
        ----------
        u : ndarray
            浓度值

        Returns
        -------
        f, df : ndarray
            反应值和导数
        """
        f = self.r * u * (1.0 - u)
        df = self.r * (1.0 - 2.0 * u)
        return f, df


class ReactionDiffusionControl:
    """
    反应-扩散最优控制问题设置.

    将物理模型转化为凸优化问题:

    min_{q}  J = 0.5 * w_track * ||u(T) - u_target||^2
                + 0.5 * w_control * gamma * ||q||^2
                + 0.5 * w_sparse * ||q||_1  (稀疏正则化, 可选)

    s.t.  M dU/dt + K U = F(U) + G q
          U(0) = U_0
          q_min <= q <= q_max

    离散化后的等式约束 (向后 Euler):
      A_k U^k = b_k + G q^k,  k = 1, ..., N_t

    其中:
      A_k = M/dt + K + diag(反应 Jacobian)
      b_k = M/dt * U^{k-1}

    将所有时间步拼接:
      A_all U_all = b_all + G_all q_all

    这是凸优化中的等式约束 Ax = b 的形式.
    """

    def __init__(self, n_spatial: int, n_time: int,
                 reaction_model: str = 'fisher'):
        self.n_spatial = n_spatial
        self.n_time = n_time
        self.n_total = n_spatial * n_time  # 状态变量总数
        self.n_control = n_spatial  # 控制变量数

        # 权重
        self.w_track = 1.0
        self.w_control = 0.01
        self.gamma = 1.0

        # 控制界
        self.q_min = -1.0
        self.q_max = 1.0

        # 反应模型
        if reaction_model == 'fisher':
            self.reaction = FisherKPPEquation()
        else:
            self.reaction = GrayScottModel()

    def build_objective_gradient(self, u_current: np.ndarray,
                                  u_target: np.ndarray,
                                  q_current: np.ndarray) -> np.ndarray:
        """
        计算目标函数梯度.

        J = 0.5 * w_track * ||u - u_target||^2 + 0.5 * w_control * gamma * ||q||^2

        nabla_q J = w_control * gamma * q  (仅控制变量的梯度)
        nabla_u J = w_track * (u - u_target)  (状态变量的梯度)

        Parameters
        ----------
        u_current : ndarray
            当前状态
        u_target : ndarray
            目标状态
        q_current : ndarray
            当前控制

        Returns
        -------
        grad : ndarray
            目标梯度 (拼接 [grad_u, grad_q])
        """
        grad_u = self.w_track * (u_current - u_target)
        grad_q = self.w_control * self.gamma * q_current
        return np.concatenate([grad_u, grad_q])

    def build_hessian_approximation(self) -> np.ndarray:
        """
        构建 Gauss-Newton Hessian 近似.

        对二次目标和线性约束:
          H = diag(w_track * I, w_control * gamma * I)

        Parameters
        ----------
        Returns
        -------
        H : ndarray
            Hessian 近似
        """
        n = self.n_total + self.n_control
        H = np.zeros((n, n))

        # 状态部分
        for i in range(self.n_total):
            H[i, i] = self.w_track

        # 控制部分
        for i in range(self.n_control):
            H[self.n_total + i, self.n_total + i] = self.w_control * self.gamma

        return H
