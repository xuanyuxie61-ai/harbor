"""
spherical_array_geometry.py
球面麦克风阵列与次级声源的几何布置

融合原始项目:
  - 1117_sphere_fibonacci_grid (球面Fibonacci均匀采样)

科学背景:
  在3D主动噪声控制中,为获得全向性声场控制,
  次级声源和误差麦克风常布置在球面上.

  Fibonacci球面网格提供近乎均匀的球面采样:
      黄金比例: phi = (1+sqrt(5))/2
      第i个点:
          theta_i = 2*pi*i/phi
          z_i = i / R
          r_i = sqrt(1 - z_i^2)
          (x,y,z) = (r_i*cos(theta_i), r_i*sin(theta_i), z_i)

  该布置优化了球谐函数展开的采样条件,
  使模态阶数达到 L_max ~ sqrt(N)/2.
"""

import numpy as np
import math


def sphere_fibonacci_grid_points(ng, radius=1.0):
    """
    生成Fibonacci球面网格点.

    参数:
        ng: 球面上点的数量
        radius: 球半径 [m]

    返回:
        points: (ng, 3) 坐标数组
    """
    if ng <= 0:
        raise ValueError("ng must be positive")

    # TODO [Hole 1]: 实现Fibonacci球面均匀采样算法
    # 要求: 使用黄金比例生成球面上近似均匀分布的ng个点
    # 提示: 需要计算每个点的角度theta、高度z、以及xy平面投影,
    #      最终返回形状为(ng, 3)的坐标数组,并乘以radius
    raise NotImplementedError("Hole 1: sphere_fibonacci_grid_points 待实现")


def spherical_harmonic_transform_matrix(points, L_max):
    """
    构造球谐函数变换矩阵 (实数形式).

    球谐函数:
        Y_l^m(\theta,\phi), l=0..L_max, m=-l..l

    对于主动噪声控制,声场可展开为:
        p(r,\theta,\phi) = \sum_{l=0}^{L} \sum_{m=-l}^{l}
                           a_l^m h_l^{(2)}(kr) Y_l^m(\theta,\phi)

    本矩阵用于从球面采样点声压估计球谐系数 a_l^m.
    """
    from scipy.special import sph_harm
    N = points.shape[0]
    n_coeffs = (L_max + 1) ** 2

    # 转为球坐标
    r = np.linalg.norm(points, axis=1)
    theta = np.arccos(np.clip(points[:, 2] / (r + 1e-12), -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])

    Y = np.zeros((N, n_coeffs), dtype=complex)
    idx = 0
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            Y[:, idx] = sph_harm(m, l, phi, theta)
            idx += 1

    # 使用实部矩阵 (对于实声场)
    return np.real(Y)


def spherical_array_directivity(weights, points, theta_grid, phi_grid, k, radius=1.0):
    """
    计算球面阵列的波束形成指向性.

    波束形成输出:
        B(\theta,\phi) = \sum_{n=1}^{N} w_n \exp(j k \hat{r}(\theta,\phi) \cdot \vec{r}_n)

    参数:
        weights: (N,) 复数权重
        points: (N,3) 阵元坐标
        theta_grid: 极角网格
        phi_grid: 方位角网格
        k: 波数
        radius: 阵列半径

    返回:
        B: 指向性函数 (二维数组)
    """
    N = points.shape[0]
    weights = np.asarray(weights, dtype=complex)

    B = np.zeros((len(theta_grid), len(phi_grid)), dtype=complex)
    for ti, th in enumerate(theta_grid):
        for pi_, ph in enumerate(phi_grid):
            r_hat = np.array([np.sin(th) * np.cos(ph),
                              np.sin(th) * np.sin(ph),
                              np.cos(th)])
            phase = k * np.dot(points, r_hat)
            B[ti, pi_] = np.sum(weights * np.exp(1j * phase))

    return np.abs(B)
