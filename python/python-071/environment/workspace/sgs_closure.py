# -*- coding: utf-8 -*-
"""
sgs_closure.py
亚格子尺度 (SGS) 闭合模型模块

融合来源:
- 860_pendulum_nonlinear_exact: 非线性动力学与能量守恒思想
- 1290_tree_chaos: 迭代函数系统 (IFS) 生成分形湍流结构

功能:
- Smagorinsky 亚格子模型
- 动态 Smagorinsky 模型（Germano 恒等式）
- 结构函数模型
- 利用 IFS 生成多尺度湍流分形结构以模拟 SGS 应力

数学背景:
  大涡模拟 (LES) 中，对 Navier-Stokes 方程进行空间滤波:
    du_i/dt + d/dx_j (u_i * u_j) = -1/rho * dp/dx_i + nu * d2u_i/dx_j2
  滤波后引入亚格子应力:
    tau_{ij} = u_i*u_j_bar - u_i_bar * u_j_bar

  Smagorinsky 模型:
    tau_{ij} - 1/3 * delta_{ij} * tau_{kk} = -2 * nu_sgs * S_{ij}
    nu_sgs = (C_s * Delta)^2 * |S|
    其中 S_{ij} = 0.5 * (du_i/dx_j + du_j/dx_i) 为滤波应变率张量
    |S| = sqrt(2 * S_{ij} * S_{ij})
    Delta 为滤波宽度，C_s ~ 0.1-0.2 为 Smagorinsky 常数

  Germano 动态模型:
    通过引入测试滤波器（宽度 hat > bar），利用 Germano 恒等式:
      L_{ij} = T_{ij} - tau_{ij}_hat = u_i*u_j_hat_bar - u_i_hat_bar * u_j_hat_bar
      L_{ij} = -2 * C * Delta^2 * (|S_hat| * S_{ij}_hat - |S|_hat * S_{ij}_hat_bar)
    通过最小二乘估计动态系数 C。
"""

import numpy as np


def strain_rate_tensor(u, v, w, dx, dy, dz):
    """
    计算三维应变率张量 S_{ij}。

    数学公式:
      S_{ij} = 0.5 * (du_i/dx_j + du_j/dx_i)

    返回:
      S11, S12, S13, S22, S23, S33: 六个独立分量
    """
    def ddx(f):
        result = np.zeros_like(f)
        result[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dx)
        return result

    def ddy(f):
        result = np.zeros_like(f)
        result[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dy)
        return result

    def ddz(f):
        result = np.zeros_like(f)
        result[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dz)
        return result

    dudx = ddx(u)
    dudy = ddy(u)
    dudz = ddz(u)
    dvdx = ddx(v)
    dvdy = ddy(v)
    dvdz = ddz(v)
    dwdx = ddx(w)
    dwdy = ddy(w)
    dwdz = ddz(w)

    S11 = dudx
    S22 = dvdy
    S33 = dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)

    return S11, S12, S13, S22, S23, S33


def smagorinsky_model(u, v, w, dx, dy, dz, Cs=0.18):
    """
    经典 Smagorinsky 亚格子粘度模型。

    参数:
      u, v, w: 滤波后的速度分量
      dx, dy, dz: 网格间距
      Cs: Smagorinsky 常数（典型值 0.1-0.2）

    返回:
      nu_sgs: 亚格子粘度场
      tau: SGS 应力张量分量
    """
    # TODO(Hole 1): 实现 Smagorinsky 亚格子粘度模型的核心计算。
    # 要求:
    #   1. 调用 strain_rate_tensor(u, v, w, dx, dy, dz) 获取应变率张量分量
    #   2. 计算滤波宽度 Delta = (dx * dy * dz)^(1/3)
    #   3. 计算应变率张量模 |S| = sqrt(2 * S_ij * S_ij)
    #   4. 计算 SGS 粘度 nu_sgs = (Cs * Delta)^2 * |S|
    #   5. 计算 SGS 应力 tau_ij = -2 * nu_sgs * S_ij
    # 返回: nu_sgs, (tau11, tau12, tau13, tau22, tau23, tau33)
    # 注意: 需要与 main.py 中调用方和 time_marching.py 中 NS 求解器的
    #       nu_eff 参数类型（标量/场）保持一致。
    raise NotImplementedError("Hole 1: Smagorinsky model core formula not implemented")


def dynamic_smagorinsky(u, v, w, dx, dy, dz, Cs_test=0.18):
    """
    动态 Smagorinsky 模型（Germano 恒等式）。

    数学模型:
      定义测试滤波器（宽度为 2*Delta），利用 Germano 恒等式:
        L_{ij} = wide{u_i*u_j} - wide{u_i}*wide{u_j}
      其中 wide{...} 表示测试滤波。

      假设:
        tau_{ij} = -2 * C * Delta^2 * |S| * S_{ij}
        T_{ij} = -2 * C * (2*Delta)^2 * |wide{S}| * wide{S}_{ij}

      Germano 恒等式:
        L_{ij} = T_{ij} - wide{tau_{ij}}
                = -2 * C * Delta^2 * M_{ij}
      其中 M_{ij} = 4*|wide{S}|*wide{S}_{ij} - wide{|S|*S_{ij}}

      通过最小二乘估计 C:
        C = -0.5 * <L_{ij} * M_{ij}> / <M_{kl} * M_{kl}>
      其中 <...> 表示局部平均（通常沿流向平均）。
    """
    # 测试滤波：简单的体积平均（2x2x2 网格）
    def test_filter(f):
        result = np.zeros_like(f)
        result[1:-1, 1:-1, 1:-1] = 0.125 * (
            f[1:-1, 1:-1, 1:-1] + f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]
            + f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1]
            + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2]
        )
        return result

    # 基础滤波应变率
    S11, S12, S13, S22, S23, S33 = strain_rate_tensor(u, v, w, dx, dy, dz)
    Delta = (dx * dy * dz) ** (1.0 / 3.0)
    S_mag = np.sqrt(2.0 * (S11 ** 2 + S22 ** 2 + S33 ** 2
                           + 2.0 * S12 ** 2 + 2.0 * S13 ** 2 + 2.0 * S23 ** 2))
    S_mag = np.clip(S_mag, 1e-10, 1e6)

    # 测试滤波速度
    u_test = test_filter(u)
    v_test = test_filter(v)
    w_test = test_filter(w)

    # 测试滤波应变率
    S11_t, S12_t, S13_t, S22_t, S23_t, S33_t = strain_rate_tensor(
        u_test, v_test, w_test, dx, dy, dz)
    S_mag_t = np.sqrt(2.0 * (S11_t ** 2 + S22_t ** 2 + S33_t ** 2
                             + 2.0 * S12_t ** 2 + 2.0 * S13_t ** 2 + 2.0 * S23_t ** 2))
    S_mag_t = np.clip(S_mag_t, 1e-10, 1e6)

    # Germano 恒等式中的 L_{ij}
    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    L11 = test_filter(uu) - u_test * u_test
    L12 = test_filter(uv) - u_test * v_test
    L13 = test_filter(uw) - u_test * w_test
    L22 = test_filter(vv) - v_test * v_test
    L23 = test_filter(vw) - v_test * w_test
    L33 = test_filter(ww) - w_test * w_test

    # M_{ij}
    M11 = 4.0 * S_mag_t * S11_t - test_filter(S_mag * S11)
    M12 = 4.0 * S_mag_t * S12_t - test_filter(S_mag * S12)
    M13 = 4.0 * S_mag_t * S13_t - test_filter(S_mag * S13)
    M22 = 4.0 * S_mag_t * S22_t - test_filter(S_mag * S22)
    M23 = 4.0 * S_mag_t * S23_t - test_filter(S_mag * S23)
    M33 = 4.0 * S_mag_t * S33_t - test_filter(S_mag * S33)

    # 最小二乘估计 C
    LM = (L11 * M11 + L12 * M12 + L13 * M13
          + L12 * M12 + L22 * M22 + L23 * M23
          + L13 * M13 + L23 * M23 + L33 * M33)
    MM = (M11 ** 2 + M12 ** 2 + M13 ** 2
          + M12 ** 2 + M22 ** 2 + M23 ** 2
          + M13 ** 2 + M23 ** 2 + M33 ** 2)

    # 边界处理
    MM = np.where(MM < 1e-15, 1e-15, MM)
    C_dynamic = -0.5 * LM / MM

    # 限制 C 的范围（避免数值不稳定）
    C_dynamic = np.clip(C_dynamic, -0.5, 0.5)

    nu_sgs = C_dynamic * Delta ** 2 * S_mag
    nu_sgs = np.clip(nu_sgs, 0.0, 10.0 * nu_sgs.max())

    return nu_sgs, C_dynamic


def ifs_turbulence_generator(n_points=5000, n_iter=1000, seed=42):
    """
    利用迭代函数系统 (IFS) 生成多尺度湍流分形结构。
    融合自 1290_tree_chaos 的 tree_chaos。

    数学模型:
      在分形湍流中，能量谱 E(k) ~ k^{-5/3} 暗示了尺度间的自相似性。
      IFS 通过仿射变换的随机组合生成分形吸引子:
        x_{n+1} = A_j * x_n + b_j
      其中 j 按概率 p_j 随机选择。

      这里构造 4 个仿射变换，模拟湍流能量级串的不同尺度:
      - 大尺度（能量注入）
      - 中尺度 1（惯性区）
      - 中尺度 2（惯性区）
      - 小尺度（能量耗散）

    参数:
      n_points: 输出点数
      n_iter: 迭代次数
      seed: 随机种子

    返回:
      points: (n_points, 3) 分形点云
      energies: 各点的能量级别
    """
    rng = np.random.default_rng(seed)

    # 4 个仿射变换（模拟湍流不同尺度）
    transforms = [
        # 大尺度 (能量注入)
        {
            'A': np.array([[0.05, 0.0, 0.0],
                           [0.0, 0.05, 0.0],
                           [0.0, 0.0, 0.05]]),
            'b': np.array([0.5, 0.0, 0.0]),
            'scale': 1.0
        },
        # 中尺度 1
        {
            'A': np.array([[0.42, -0.42, 0.0],
                           [0.42, 0.42, 0.0],
                           [0.0, 0.0, 0.1]]),
            'b': np.array([0.29, -0.01, 0.0]),
            'scale': 0.5
        },
        # 中尺度 2
        {
            'A': np.array([[0.42, 0.42, 0.0],
                           [-0.42, 0.42, 0.0],
                           [0.0, 0.0, 0.1]]),
            'b': np.array([0.29, 0.41, 0.0]),
            'scale': 0.5
        },
        # 小尺度 (能量耗散)
        {
            'A': np.array([[0.1, 0.0, 0.0],
                           [0.0, 0.1, 0.0],
                           [0.0, 0.0, 0.1]]),
            'b': np.array([0.45, 0.15, 0.0]),
            'scale': 0.1
        }
    ]

    # 随机起点
    x = rng.random(3)
    points = []
    energies = []

    # 预热
    for _ in range(100):
        j = rng.integers(0, 4)
        x = transforms[j]['A'] @ x + transforms[j]['b']

    # 主迭代
    for _ in range(n_iter):
        j = rng.integers(0, 4)
        x = transforms[j]['A'] @ x + transforms[j]['b']
        points.append(x.copy())
        energies.append(transforms[j]['scale'])

    points = np.array(points)
    energies = np.array(energies)

    # 均匀采样 n_points
    if len(points) > n_points:
        idx = rng.choice(len(points), size=n_points, replace=False)
        points = points[idx]
        energies = energies[idx]

    return points, energies


def structure_function_model(u, v, w, dx, dy, dz, order=2):
    """
    结构函数亚格子模型。

    数学模型:
      二阶结构函数:
        D_{ij}(r) = <(u_i(x+r) - u_i(x)) * (u_j(x+r) - u_j(x))>
      SGS 粘度:
        nu_sgs = C_{SF} * Delta * sqrt(D_{LL}(Delta))
      其中 D_{LL} 为纵向结构函数。
    """
    # 计算局部结构函数（简化：仅考虑相邻网格点）
    du_x = u[1:, :, :] - u[:-1, :, :]
    dv_y = v[:, 1:, :] - v[:, :-1, :]
    dw_z = w[:, :, 1:] - w[:, :, :-1]

    # 纵向结构函数（近似）
    D_ll = np.zeros_like(u)
    D_ll[1:-1, 1:-1, 1:-1] = 0.333 * (
        (du_x[1:, 1:-1, 1:-1] ** 2 + du_x[:-1, 1:-1, 1:-1] ** 2)
        + (dv_y[1:-1, 1:, 1:-1] ** 2 + dv_y[1:-1, :-1, 1:-1] ** 2)
        + (dw_z[1:-1, 1:-1, 1:] ** 2 + dw_z[1:-1, 1:-1, :-1] ** 2)
    )

    Delta = (dx * dy * dz) ** (1.0 / 3.0)
    C_SF = 1.4

    D_ll = np.clip(D_ll, 0.0, 1e6)
    nu_sgs = C_SF * Delta * np.sqrt(D_ll)

    return nu_sgs
