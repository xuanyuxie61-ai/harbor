"""
physics_constants.py

电磁学基本物理常数与麦克斯韦方程组的数学表达。
本模块定义了真空中的电磁参数以及时域麦克斯韦方程组的离散化形式。

核心物理模型:
--------------
1. 麦克斯韦-安培定律（微分形式）:
   ∇ × H = J + ∂D/∂t
   其中 D = εE, J = σE
   => ∇ × H = σE + ε ∂E/∂t

2. 麦克斯韦-法拉第定律（微分形式）:
   ∇ × E = -∂B/∂t
   其中 B = μH
   => ∇ × E = -μ ∂H/∂t

3. 三维旋度算子:
   ∇ × F = (∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y)

4. 电磁能量密度:
   w = ½(ε|E|² + μ|H|²)

5. 坡印廷矢量（能流密度）:
   S = E × H

6. 谐振腔品质因数:
   Q = ω · (储能) / (损耗功率) = ωW/P_loss
"""

import numpy as np

# 真空物理常数 (SI单位制)
MU_0 = 4.0 * np.pi * 1e-7          # 真空磁导率 [H/m]
EPSILON_0 = 8.854187817e-12        # 真空介电常数 [F/m]
C_0 = 1.0 / np.sqrt(MU_0 * EPSILON_0)  # 真空中光速 [m/s]
ETA_0 = np.sqrt(MU_0 / EPSILON_0)  # 真空波阻抗 [Ω]


def curl_electric_to_magnetic(E, dx, dy, dz):
    """
    计算电场旋度，用于更新磁场（法拉第定律）。

    离散化格式（中心差分）:
    (∇ × E)_x|_{i,j+½,k+½} ≈ (Ez_{i,j+1,k+½} - Ez_{i,j,k+½})/dy
                              - (Ey_{i,j+½,k+1} - Ey_{i,j+½,k})/dz

    Parameters
    ----------
    E : tuple of ndarray
        (Ex, Ey, Ez) 电场分量，每个为三维数组
    dx, dy, dz : float
        网格步长

    Returns
    -------
    tuple of ndarray
        (∇×E)_x, (∇×E)_y, (∇×E)_z
    """
    Ex, Ey, Ez = E

    # TODO: Hole 1 — 实现电场旋度的三个分量离散计算
    # 旋度的x分量: ∂Ez/∂y - ∂Ey/∂z
    # 需要对内部点使用中心差分，边界使用前向/后向差分
    # 旋度的y分量: ∂Ex/∂z - ∂Ez/∂x
    # 旋度的z分量: ∂Ey/∂x - ∂Ex/∂y
    raise NotImplementedError("Hole 1: 请实现curl_electric_to_magnetic的旋度离散计算")


def curl_magnetic_to_electric(H, dx, dy, dz):
    """
    计算磁场旋度，用于更新电场（安培定律）。

    离散化格式与curl_electric类似，但交错位置不同。
    """
    Hx, Hy, Hz = H

    # 旋度的x分量: ∂Hz/∂y - ∂Hy/∂z
    dHz_dy = np.zeros_like(Hx)
    dHy_dz = np.zeros_like(Hx)
    dHz_dy[:, :-1, :] = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy
    dHy_dz[:, :, :-1] = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz
    curl_x = dHz_dy - dHy_dz

    # 旋度的y分量: ∂Hx/∂z - ∂Hz/∂x
    dHx_dz = np.zeros_like(Hy)
    dHz_dx = np.zeros_like(Hy)
    dHx_dz[:, :, :-1] = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz
    dHz_dx[:-1, :, :] = (Hz[1:, :, :] - Hz[:-1, :, :]) / dx
    curl_y = dHx_dz - dHz_dx

    # 旋度的z分量: ∂Hy/∂x - ∂Hx/∂y
    dHy_dx = np.zeros_like(Hz)
    dHx_dy = np.zeros_like(Hz)
    dHy_dx[:-1, :, :] = (Hy[1:, :, :] - Hy[:-1, :, :]) / dx
    dHx_dy[:, :-1, :] = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy
    curl_z = dHy_dx - dHx_dy

    return curl_x, curl_y, curl_z


def electromagnetic_energy_density(E, H, epsilon, mu):
    """
    计算电磁能量密度 w = ½(ε|E|² + μ|H|²)

    Parameters
    ----------
    E, H : tuple of ndarray
        电场和磁场分量
    epsilon, mu : ndarray
        介电常数和磁导率分布

    Returns
    -------
    ndarray
        能量密度
    """
    Ex, Ey, Ez = E
    Hx, Hy, Hz = H
    E_magnitude_sq = Ex**2 + Ey**2 + Ez**2
    H_magnitude_sq = Hx**2 + Hy**2 + Hz**2
    return 0.5 * (epsilon * E_magnitude_sq + mu * H_magnitude_sq)


def poynting_vector(E, H):
    """
    计算坡印廷矢量 S = E × H

    Returns
    -------
    tuple of ndarray
        Sx, Sy, Sz
    """
    Ex, Ey, Ez = E
    Hx, Hy, Hz = H
    Sx = Ey * Hz - Ez * Hy
    Sy = Ez * Hx - Ex * Hz
    Sz = Ex * Hy - Ey * Hx
    return Sx, Sy, Sz


def quality_factor(omega, W_stored, P_loss, eps=1e-30):
    """
    计算谐振腔品质因数 Q = ω · W_stored / P_loss

    Parameters
    ----------
    omega : float
        角频率 [rad/s]
    W_stored : float
        总储能 [J]
    P_loss : float
        损耗功率 [W]
    eps : float
        数值稳定性保护

    Returns
    -------
    float
        品质因数Q
    """
    if P_loss < eps:
        return float('inf')
    return omega * W_stored / P_loss


def cfl_condition_3d(dx, dy, dz, c_max):
    """
    三维CFL稳定性条件:
    Δt ≤ 1 / (c · sqrt(1/Δx² + 1/Δy² + 1/Δz²))

    Parameters
    ----------
    dx, dy, dz : float
        网格步长
    c_max : float
        最大波速

    Returns
    -------
    float
        最大允许时间步长
    """
    return 1.0 / (c_max * np.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))


def wavenumber_frequency_relation(omega, epsilon, mu):
    """
    色散关系: k² = ω²με

    Parameters
    ----------
    omega : float
        角频率
    epsilon, mu : float
        介质参数

    Returns
    -------
    float
        波数k
    """
    return omega * np.sqrt(epsilon * mu)
