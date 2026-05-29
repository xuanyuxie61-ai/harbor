"""
multimode_coupling.py
多模光纤中的模式耦合轨道动力学

融合原项目:
  - 345_exm/orbits: N 体引力轨道动力学积分

科学背景:
  在多模光纤或双折射光纤中，不同模式（或偏振态）之间存在线性耦合
  和非线性交叉相位调制（XPM）。当模式数 M 较大时，模式间的能量交换
  类似于 N 体引力系统：每个模式感受来自其他所有模式的"力"（耦合项）。

  模式耦合的广义 Hamilton 系统可写为:
      i * dA_m/dz = sum_n K_{mn} A_n + gamma * sum_{n,p,q} chi_{mnpq} A_n A_p^* A_q
  其中 K_{mn} 为线性耦合矩阵，chi_{mnpq} 为非线性耦合张量。

  在简化模型中（只考虑双模或三模耦合），系统退化为耦合非线性
  Schrodinger 方程（CNLSE），可用类似于 N 体轨道的辛积分方法求解。

  本模块将 N 体引力动力学的 Verlet/辛积分框架移植到模式耦合问题，
  追踪各模式功率随传播距离的"轨道"。
"""

import numpy as np
from typing import Tuple


def mode_coupling_matrix(n_modes: int, delta_beta: float,
                         coupling_coeff: float) -> np.ndarray:
    """
    构造简化的模式耦合矩阵 K_{mn}。

    对于相邻模式，耦合系数为 kappa；对角元为各模式的传播常数差 delta_beta_m。

    公式:
        K_{mm} = delta_beta * (m - (n_modes-1)/2)
        K_{m,m+1} = K_{m+1,m} = coupling_coeff

    Parameters
    ----------
    n_modes : int
        模式数量。
    delta_beta : float
        相邻模式传播常数差（1/m）。
    coupling_coeff : float
        线性耦合系数（1/m）。

    Returns
    -------
    np.ndarray
        耦合矩阵，形状 (n_modes, n_modes)。
    """
    K = np.zeros((n_modes, n_modes), dtype=complex)
    for m in range(n_modes):
        K[m, m] = delta_beta * (m - (n_modes - 1) / 2.0)
        if m + 1 < n_modes:
            K[m, m + 1] = coupling_coeff
            K[m + 1, m] = coupling_coeff
    return K


def xpm_coefficients(n_modes: int, gamma: float,
                      overlap_factors: np.ndarray) -> np.ndarray:
    """
    计算交叉相位调制（XPM）系数张量。

    简化模型中，chi_{mnpq} 非零仅当 m=q 且 n=p（相位匹配条件）。
    自相位调制（SPM）系数: chi_{mmmm} = gamma * O_{mmmm}
    交叉相位调制系数: chi_{mnnm} = 2*gamma * O_{mnnm}  (m != n)

    对于简并模式，通常 O_{mmmm} = 1, O_{mnnm} ~ 1/2。

    Parameters
    ----------
    n_modes : int
        模式数。
    gamma : float
        非线性系数。
    overlap_factors : np.ndarray
        重叠积分因子，形状 (n_modes, n_modes)。

    Returns
    -------
    np.ndarray
        chi 张量的对角/非对角部分，形状 (n_modes, n_modes)。
    """
    chi = np.zeros((n_modes, n_modes), dtype=float)
    for m in range(n_modes):
        for n in range(n_modes):
            if m == n:
                chi[m, n] = gamma * overlap_factors[m, n]
            else:
                chi[m, n] = 2.0 * gamma * overlap_factors[m, n]
    return chi


def multimode_propagation_verlet(A0: np.ndarray, z_target: float,
                                  K: np.ndarray, chi: np.ndarray,
                                  dz: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用辛积分（Verlet 类方法）求解多模耦合方程。

    将模式振幅的演化类比为 N 体轨道问题：
        dA_m/dz = -i * sum_n K_{mn} A_n - i * gamma * sum_n chi_{mn} |A_n|^2 * A_m

    这类似于 Hamilton 系统中的运动方程:
        dq/dt = p
        dp/dt = F(q)
    其中将实部和虚部分离后可用辛积分保持能量守恒。

    本实现采用分步辛方法：
        线性半步:  exp(-i*K*dz/2)
        非线性全步（类似质点轨道推进）:  A_m <- A_m * exp(-i*gamma*chi_{mm}*|A_m|^2*dz)
        线性半步:  exp(-i*K*dz/2)

    Parameters
    ----------
    A0 : np.ndarray
        初始模式振幅，形状 (n_modes,)。
    z_target : float
        总传播距离。
    K : np.ndarray
        耦合矩阵。
    chi : np.ndarray
        XPM 系数矩阵。
    dz : float
        步长。

    Returns
    -------
    tuple
        (z_array, A_history) 其中 A_history 形状 (n_steps, n_modes)。
    """
    n_modes = len(A0)
    n_steps = int(np.ceil(z_target / dz)) + 1
    z_array = np.linspace(0.0, z_target, n_steps)
    A_history = np.zeros((n_steps, n_modes), dtype=complex)
    A = A0.copy()
    A_history[0, :] = A
    # 预计算线性半步传播算子
    exp_K_half = np.linalg.matrix_power(np.eye(n_modes, dtype=complex), 1)
    # 更准确地: expm(-i*K*dz/2)
    from scipy.linalg import expm
    U_half = expm(-1j * K * dz * 0.5)
    for step in range(1, n_steps):
        # 线性半步
        A = U_half @ A
        # 非线性全步（对角相位）
        for m in range(n_modes):
            phi = 0.0
            for n in range(n_modes):
                phi += chi[m, n] * abs(A[n]) ** 2
            A[m] *= np.exp(-1j * phi * dz)
        # 线性半步
        A = U_half @ A
        A_history[step, :] = A
    return z_array, A_history


def mode_power_orbits(A_history: np.ndarray) -> np.ndarray:
    """
    从模式振幅历史提取各模式功率的"轨道"。

    类比天体轨道，模式功率 P_m(z) = |A_m(z)|^2 的演化轨迹可视为
    在 M 维功率空间中的轨道。

    Parameters
    ----------
    A_history : np.ndarray
        振幅历史，形状 (n_steps, n_modes)。

    Returns
    -------
    np.ndarray
        功率历史，形状 (n_steps, n_modes)。
    """
    return np.abs(A_history) ** 2


def orbital_angular_momentum_modes(l_values: np.ndarray,
                                    r_grid: np.ndarray,
                                    phi_grid: np.ndarray) -> np.ndarray:
    """
    生成轨道角动量（OAM）模式的横向场分布。

    OAM 模式的螺旋相位结构:
        E_{l,p}(r, phi) = u_{l,p}(r) * exp(i*l*phi)

    其中 l 为拓扑荷数，p 为径向节点数。
    在弱导近似下，u_{l,p}(r) 可用 Laguerre-Gaussian 函数近似:
        u_{l,p}(r) ~ (sqrt(2)*r/w0)^{|l|} * L_p^{|l|}(2*r^2/w0^2) *
                     exp(-r^2/w0^2)

    OAM 模式在多模光纤中的耦合是超连续谱空间结构调控的前沿方向。

    Parameters
    ----------
    l_values : np.ndarray
        拓扑荷数组。
    r_grid : np.ndarray
        径向坐标。
    phi_grid : np.ndarray
        角向坐标。

    Returns
    -------
    np.ndarray
        模式场，形状 (n_l, n_r, n_phi)。
    """
    from scipy.special import genlaguerre
    n_l = len(l_values)
    n_r = len(r_grid)
    n_phi = len(phi_grid)
    modes = np.zeros((n_l, n_r, n_phi), dtype=complex)
    w0 = np.max(r_grid) / 2.0
    for il, l in enumerate(l_values):
        for ir, r in enumerate(r_grid):
            rho = np.sqrt(2.0) * r / w0
            radial = (rho ** abs(l)) * genlaguerre(0, abs(l))(rho ** 2) * np.exp(-rho ** 2 / 2.0)
            for ip, phi in enumerate(phi_grid):
                modes[il, ir, ip] = radial * np.exp(1j * l * phi)
    return modes
