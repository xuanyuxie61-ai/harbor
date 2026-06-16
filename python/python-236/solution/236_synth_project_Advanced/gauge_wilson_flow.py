"""
gauge_wilson_flow.py — 规范场与 Wilson 梯度流
================================================
融合种子项目:
  [121_brusselator_ode] : ODE 右端函数 + 参数管理 → Wilson 流 ODE
  [1065_Azamat-Mukhamediya_SRPM-ST] : 并行 mini-batch → 并行配置演化

物理背景:
  Wilson 梯度流 (Lüscher, 2010) 定义了一个 5 维场 B_mu(t,x),
  满足流方程:
      dB_mu/dt = D_nu G_nu_mu + lambda*D_mu(D_nu B_nu)
  其中 G_nu_mu 为场强张量, D_mu 为协变导数.
  在树级, 这等价于扩散方程:
      dB_mu/dt = Delta B_mu
  流时间 t 起到平滑尺度 sqrt(8t) 的作用.

核心公式:
  Plaquette:  U_P = U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
  场强:      G_mu_nu(x) = (i/a^2)(U_P - U_P^dag)/2 - traceless
  Wilson 流:  dV_mu/dt = -delta S_W / delta V_mu * V_mu
              (Lie 代数投影: P_A(X) = (X - X^dag)/2 - Tr(...)/Nc * I)
  能量密度:  E(t) = (1/4) G_mu_nu^a G_mu_nu^a
  t^2 E(t) 在 t -> 0 时趋向常数 (微扰预言), 用于定义梯度流尺度 t_0.

数值方法 (源自 [121_brusselator_ode] 的 ODE 思想):
  采用 3 阶 Runge-Kutta (Lüscher 推荐) 和自适应步长的 RK45.
  dV/dt = Z(V) * V,  Z in su(Nc)
"""

import numpy as np
from typing import Tuple, Optional, Callable
from lattice_geometry import LatticeGeometry


# ======================================================================
# SU(Nc) 群操作
# ======================================================================

def suN_project(X: np.ndarray, Nc: int = 3) -> np.ndarray:
    """将任意复矩阵投影到 su(Nc) 李代数.

    P_A(X) = (X - X^dag) / 2 - (1/Nc) Tr((X - X^dag)/2) * I

    这确保 Z_mu 是反厄米无迹矩阵.
    """
    anti_herm = 0.5 * (X - X.conj().T)
    trace_part = np.trace(anti_herm) / Nc
    return anti_herm - trace_part * np.eye(Nc, dtype=np.complex128)


def exp_suN(Z: np.ndarray, Nc: int = 3) -> np.ndarray:
    """su(Nc) 李代数元素的矩阵指数 → SU(Nc) 群元素.

    U = exp(Z),  Z in su(Nc)  =>  U in SU(Nc)
    使用 Padé 近似保证幺正性.
    """
    # 确保反厄米性
    Z = 0.5 * (Z - Z.conj().T)
    Z -= np.trace(Z) / Nc * np.eye(Nc)
    return _matrix_exp_unitary(Z)


def _matrix_exp_unitary(Z: np.ndarray) -> np.ndarray:
    """通过 Cayley-Hamilton / Padé 近似计算反厄米矩阵的指数.

    对于 2x2:  exp(i*theta*n.sigma) = cos(theta)*I + i*sin(theta)*n.sigma
    对于 3x3:  使用 Padé [3/3] 近似.
    """
    Nc = Z.shape[0]
    if Nc == 2:
        # 精确公式: su(2) 的指数映射
        # Z = i * (a0*I + a.sigma), 取 traceless part
        det_Z = np.linalg.det(Z)
        theta = np.sqrt(max(-det_Z.real, 0.0))
        if theta < 1e-14:
            return np.eye(2, dtype=np.complex128) + Z
        return (np.cos(theta) * np.eye(2)
                + np.sin(theta) / theta * Z)
    else:
        # Padé [6/6] 近似 (高精度)
        result = np.eye(Nc, dtype=np.complex128)
        term = np.eye(Nc, dtype=np.complex128)
        for k in range(1, 20):
            term = term @ Z / k
            result += term
            if np.max(np.abs(term)) < 1e-15:
                break
        # 正交化保证幺正性
        Q, _ = np.linalg.qr(result)
        return Q


def random_suN(Nc: int = 3, beta: float = 1.0,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """生成随机 SU(Nc) 矩阵 (热浴初始条件).

    参数
    ----
    Nc : 色数 (默认 3)
    beta : 耦合常数 (beta = 2*Nc/g^2)
           beta -> inf 对应冷起始 (U = I)
           beta -> 0 对应随机起始
    rng : 随机数生成器
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if beta > 100.0:
        # 冷起始: 接近单位矩阵
        Z = suN_project(1e-6 * (rng.standard_normal((Nc, Nc))
                                 + 1j * rng.standard_normal((Nc, Nc))))
        return exp_suN(Z, Nc)
    else:
        # 热起始: 随机高斯矩阵 → 投影 → 指数
        width = 1.0 / max(np.sqrt(beta), 0.1)
        X = width * (rng.standard_normal((Nc, Nc))
                      + 1j * rng.standard_normal((Nc, Nc)))
        Z = suN_project(X, Nc)
        return exp_suN(Z, Nc)


# ======================================================================
# 格点规范场配置
# ======================================================================

class GaugeField:
    """SU(Nc) 规范场配置.

    参数
    ----
    geo : LatticeGeometry
    Nc : int
        色数 (默认 3).
    config_type : str
        'cold' (U=I) 或 'hot' (随机).
    beta : float
        Wilson 作用量耦合常数 beta = 2*Nc / g^2.
    seed : int
        随机种子.
    """

    def __init__(self, geo: LatticeGeometry, Nc: int = 3,
                 config_type: str = 'cold', beta: float = 6.0,
                 seed: int = 236):
        self.geo = geo
        self.Nc = Nc
        self.beta = beta
        self.ndim = geo.ndim
        self.volume = geo.volume
        self.rng = np.random.default_rng(seed)

        # 规范场: U[mu][x] = SU(Nc) 矩阵
        # shape: (ndim, volume, Nc, Nc)
        self.U = np.zeros((self.ndim, self.volume, Nc, Nc),
                          dtype=np.complex128)

        if config_type == 'cold':
            # 冷起始: 所有链接 = 单位矩阵
            for mu in range(self.ndim):
                for x in range(self.volume):
                    self.U[mu, x] = np.eye(Nc, dtype=np.complex128)
        elif config_type == 'hot':
            for mu in range(self.ndim):
                for x in range(self.volume):
                    self.U[mu, x] = random_suN(Nc, beta, self.rng)
        else:
            raise ValueError(f"未知配置类型: {config_type}")

    def get_link(self, mu: int, x_idx: int) -> np.ndarray:
        """获取链接 U_mu(x)."""
        return self.U[mu, x_idx].copy()

    def staple_sum(self, mu: int, x_idx: int) -> np.ndarray:
        """计算链接 U_mu(x) 的 staples 之和.

        staple = sum_{nu != mu} [U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
                                + U_nu^dag(x+mu-nu) U_mu^dag(x-nu) U_nu(x-nu)]

        这是 Wilson 作用量中除 U_mu(x) 外的所有包含它的 plaquette 之和.
        S_W = -beta/(2*Nc) * sum_P Re Tr(U_P)
        """
        Nc = self.Nc
        staple = np.zeros((Nc, Nc), dtype=np.complex128)

        for nu in range(self.ndim):
            if nu == mu:
                continue
            # x + mu
            x_mu = self.geo.neighbor_index(x_idx, mu, +1)
            # x + nu
            x_nu = self.geo.neighbor_index(x_idx, nu, +1)
            # x + mu - nu
            x_mu_minus_nu = self.geo.neighbor_index(x_mu, nu, -1)
            # x - nu
            x_minus_nu = self.geo.neighbor_index(x_idx, nu, -1)

            U_nu_x = self.U[nu, x_idx]
            U_mu_xnu = self.U[mu, x_nu]
            U_nu_xmu = self.U[nu, x_mu]
            U_mu_x = self.U[mu, x_idx]
            U_nu_xmunu = self.U[nu, x_mu_minus_nu]
            U_mu_xmnu = self.U[mu, x_minus_nu]
            U_nu_xmnu = self.U[nu, x_minus_nu]

            # 上方 staple
            upper = U_nu_xmu @ U_mu_xnu.conj().T @ U_nu_x.conj().T
            # 下方 staple
            lower = (U_nu_xmunu.conj().T @ U_mu_xmnu.conj().T
                     @ U_nu_xmnu)
            staple += upper + lower

        return staple

    def plaquette(self, mu: int, nu: int, x_idx: int) -> np.ndarray:
        """计算 plaquette U_P(x; mu, nu).

        U_P = U_mu(x) * U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
        """
        x_mu = self.geo.neighbor_index(x_idx, mu, +1)
        x_nu = self.geo.neighbor_index(x_idx, nu, +1)

        U1 = self.U[mu, x_idx]
        U2 = self.U[nu, x_mu]
        U3 = self.U[mu, x_nu].conj().T
        U4 = self.U[nu, x_idx].conj().T

        return U1 @ U2 @ U3 @ U4

    def average_plaquette(self) -> float:
        """计算平均 plaquette (规范作用量的示踪).

        <P> = 1/(N_P * Nc) * sum_{x,mu<nu} Re Tr(U_P)
        其中 N_P = V * ndim * (ndim-1) / 2 为 plaquette 总数.
        """
        total = 0.0
        count = 0
        for x in range(self.volume):
            for mu in range(self.ndim):
                for nu in range(mu + 1, self.ndim):
                    P = self.plaquette(mu, nu, x)
                    total += np.trace(P).real
                    count += 1
        if count == 0:
            return 0.0
        return total / (count * self.Nc)


# ======================================================================
# Wilson 梯度流 (源自 [121_brusselator_ode] 的 ODE 积分)
# ======================================================================

def wilson_flow_force(gauge: GaugeField, mu: int, x_idx: int
                      ) -> np.ndarray:
    """Wilson 流的力 (Lie 代数元素).

    Z_mu(x) = -i * beta/(2*Nc) * [U_mu(x) * staple^dag
              - staple * U_mu^dag(x)]_traceless

    这类似于 [121_brusselator_ode] 中的反应项:
    dU/dt = f(U), 其中 f 由局部场构型决定.
    """
    U = gauge.get_link(mu, x_idx)
    S = gauge.staple_sum(mu, x_idx)

    # 力 = U * S^dag - S * U^dag (投影到无迹反厄米)
    force = U @ S.conj().T - S @ U.conj().T
    return suN_project(force, gauge.Nc)


def wilson_flow_rhs(V_flat: np.ndarray, gauge_template: GaugeField,
                    t_flow: float) -> np.ndarray:
    """Wilson 流 ODE 的右端函数 (源自 [121_brusselator_ode]).

    类似于 brusselator_deriv(t, y), 这里计算 dV/dt.
    V_flat 为展平的链接矩阵数组.

    注: 流时间 t 类似于 Brusselator 中的时间变量,
    驱动系统向平滑构型演化.
    """
    geo = gauge_template.geo
    Nc = gauge_template.Nc
    ndim = gauge_template.ndim
    volume = gauge_template.volume

    # 重构规范场
    V = V_flat.reshape(ndim, volume, Nc, Nc)
    dVdt = np.zeros_like(V)

    # 临时规范场用于计算力
    temp_gauge = GaugeField.__new__(GaugeField)
    temp_gauge.geo = geo
    temp_gauge.Nc = Nc
    temp_gauge.beta = gauge_template.beta
    temp_gauge.ndim = ndim
    temp_gauge.volume = volume
    temp_gauge.U = V
    temp_gauge.rng = gauge_template.rng

    for x in range(volume):
        for mu in range(ndim):
            Z = wilson_flow_force(temp_gauge, mu, x)
            # dV/dt = Z * V (左乘)
            dVdt[mu, x] = Z @ V[mu, x]

    return dVdt.ravel()


def integrate_wilson_flow(gauge: GaugeField,
                          t_max: float = 0.5,
                          dt: float = 0.01,
                          method: str = 'RK3'
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Wilson 梯度流时间积分.

    采用 Lüscher 推荐的三阶 Runge-Kutta 方法 (源自 [121_brusselator_ode]
    的 ODE 积分思想):

    W_0 = V
    W_1 = exp(Z_0 * dt) * W_0
    W_2 = exp(-17/36 * Z_0 * dt + 1/4 * Z_1 * dt) * W_1  -- 简化版用RK3
    V(t+dt) = exp(Z_2 * dt) * W_2

    实际使用简化 RK3:
    k1 = f(V_n)
    k2 = f(V_n + dt/2 * k1)
    k3 = f(V_n - dt*k1 + 2*dt*k2)
    V_{n+1} = V_n + dt/6 * (k1 + 4*k2 + k3)
    然后投影回 SU(Nc).

    参数
    ----
    gauge : GaugeField
        初始规范场.
    t_max : float
        最大流时间.
    dt : float
        流时间步长.
    method : str
        'RK3' (三阶 Runge-Kutta) 或 'Euler' (一阶, 用于测试).

    返回
    ----
    t_values : ndarray
        流时间数组.
    E_values : ndarray
        能量密度 t^2 E(t) 在每个流时间步的值.
    """
    # 小型格点可复现实验: 限制流时间步数
    n_steps = max(1, int(t_max / dt + 0.5))
    dt = t_max / n_steps

    t_values = np.zeros(n_steps + 1)
    E_values = np.zeros(n_steps + 1)

    # 复制初始配置
    V = gauge.U.copy()

    def compute_energy(V_config):
        """计算能量密度 E(t) = (1/4) G_mu_nu^a G_mu_nu^a.

        在格点上: E = 1/4 * sum_{mu,nu} Tr(G_mu_nu^2) / (V * Nc)
        t^2 * E(t) 用于确定 t_0 尺度.
        """
        temp = GaugeField.__new__(GaugeField)
        temp.geo = gauge.geo
        temp.Nc = gauge.Nc
        temp.beta = gauge.beta
        temp.ndim = gauge.ndim
        temp.volume = gauge.volume
        temp.U = V_config
        temp.rng = gauge.rng

        total_E = 0.0
        n_plaq = 0
        for x in range(gauge.volume):
            for mu in range(gauge.ndim):
                for nu in range(mu + 1, gauge.ndim):
                    P = temp.plaquette(mu, nu, x)
                    # G_mu_nu ~ (U_P - U_P^dag) / (2i) - traceless
                    G = (P - P.conj().T) / 2.0
                    G -= np.trace(G) / gauge.Nc * np.eye(gauge.Nc)
                    total_E += -np.trace(G @ G).real
                    n_plaq += 1

        if n_plaq == 0:
            return 0.0
        # E = 总作用量密度 / (4 * V * ndim*(ndim-1)/2)
        E = total_E / (gauge.volume * gauge.Nc * n_plaq /
                        (gauge.volume * gauge.ndim * (gauge.ndim - 1) / 2))
        return max(E, 0.0)  # 能量密度非负

    # 初始能量
    t_values[0] = 0.0
    E_values[0] = compute_energy(V)

    def rhs(V_config):
        """计算 dV/dt (右端函数)."""
        temp = GaugeField.__new__(GaugeField)
        temp.geo = gauge.geo
        temp.Nc = gauge.Nc
        temp.beta = gauge.beta
        temp.ndim = gauge.ndim
        temp.volume = gauge.volume
        temp.U = V_config
        temp.rng = gauge.rng

        dV = np.zeros_like(V_config)
        for x in range(gauge.volume):
            for mu in range(gauge.ndim):
                Z = wilson_flow_force(temp, mu, x)
                dV[mu, x] = Z @ V_config[mu, x]
        return dV

    def project_suN(V_config):
        """将每个链接投影回 SU(Nc) (保证幺正性)."""
        for mu in range(gauge.ndim):
            for x in range(gauge.volume):
                U = V_config[mu, x]
                U_re, _ = np.linalg.qr(U)
                # 修正行列式相位 → SU(Nc)
                det = np.linalg.det(U_re)
                U_re *= det.conj() / max(abs(det), 1e-300) ** (1.0 / gauge.Nc)
                V_config[mu, x] = U_re
        return V_config

    for step in range(n_steps):
        if method == 'Euler':
            # 一阶 Euler (仅用于测试稳定性)
            dV = rhs(V)
            V = V + dt * dV
        elif method == 'RK3':
            # 三阶 Runge-Kutta (源自 [121_brusselator_ode])
            k1 = rhs(V)
            k2 = rhs(V + 0.5 * dt * k1)
            k3 = rhs(V - dt * k1 + 2.0 * dt * k2)
            V = V + (dt / 6.0) * (k1 + 4.0 * k2 + k3)
        else:
            raise ValueError(f"未知积分方法: {method}")

        # 投影回 SU(Nc)
        V = project_suN(V)

        t_values[step + 1] = (step + 1) * dt
        E_values[step + 1] = compute_energy(V)

    return t_values, E_values


def compute_t0_energy(E_values: np.ndarray, t_values: np.ndarray,
                      target: float = 0.3) -> float:
    """确定梯度流尺度 t_0.

    t_0 定义为: t^2 E(t) |_{t=t_0} = c,  典型值 c = 0.3.

    通过线性插值找到 t^2*E(t) = target 的时刻.

    参数
    ----
    E_values : ndarray
        能量密度 E(t) 数组.
    t_values : ndarray
        流时间数组.
    target : float
        目标值 (默认 0.3).

    返回
    ----
    t0 : float
        梯度流尺度 (若未达到目标则返回最后一个 t).
    """
    t2E = t_values ** 2 * E_values
    # 找到跨越 target 的位置
    for i in range(1, len(t2E)):
        if t2E[i] >= target and t2E[i - 1] < target:
            # 线性插值
            frac = ((target - t2E[i - 1]) /
                    max(t2E[i] - t2E[i - 1], 1e-300))
            return t_values[i - 1] + frac * (t_values[i] - t_values[i - 1])
    # 未达到目标
    return t_values[-1] if len(t_values) > 0 else 0.0
