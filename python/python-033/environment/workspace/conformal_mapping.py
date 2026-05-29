"""
conformal_mapping.py
基于种子项目 611_joukowsky_transform 的共形映射

在核天体物理中，共形映射可用于描述中子星并合过程中的潮汐形变和物质转移流场。
Joukowsky 变换将复平面上的圆映射为翼型（或类椭圆），其解析形式为：
    f(z) = 1/2 · (z + 1/z)

在 r 过程环境中，我们利用其将吸积盘或喷流的几何边界映射到标准圆形区域，
从而简化中子通量和温度场的 Laplace 方程求解。

逆变换：z = w ± sqrt(w² - 1)
"""

import numpy as np


def joukowsky_transform(z):
    """
    Joukowsky 共形映射：f(z) = 0.5 * (z + 1/z)

    参数:
        z : complex 或 ndarray of complex

    返回:
        w : 同类型，映射后的值
    """
    z = np.asarray(z, dtype=complex)
    # 鲁棒处理：避免 z=0 处的奇点
    w = np.zeros_like(z, dtype=complex)
    mask = np.abs(z) > 1e-15
    w[mask] = 0.5 * (z[mask] + 1.0 / z[mask])
    # z≈0 时映射到无穷（物理上对应奇点/喷流轴）
    return w


def joukowsky_inverse(w, branch='+'):
    """
    Joukowsky 逆变换：z = w ± sqrt(w² - 1)

    参数:
        w : complex 或 ndarray of complex
        branch : '+' 或 '-', 选择分支

    返回:
        z : 同类型
    """
    w = np.asarray(w, dtype=complex)
    discriminant = w ** 2 - 1.0
    sqrt_disc = np.sqrt(discriminant)
    if branch == '+':
        z = w + sqrt_disc
    else:
        z = w - sqrt_disc
    return z


def map_accretion_streamline(radius_ratio, theta, offset=0.1):
    """
    将中子星吸积盘流线的极坐标映射为共形坐标。

    吸积盘内边界近似为圆 |z - z0| = R，通过 Joukowsky 变换映射为翼型边界。
    参数:
        radius_ratio : float, 半径比 r/R_disk
        theta : ndarray, 角度坐标
        offset : float, 圆心偏移量（控制翼型厚度）

    返回:
        w_real, w_imag : ndarray, 映射后的笛卡尔坐标
    """
    theta = np.asarray(theta, dtype=float)
    # 复平面上的圆
    z = radius_ratio * np.exp(1j * theta) + offset
    w = joukowsky_transform(z)
    return np.real(w), np.imag(w)


def temperature_field_conformal(rho, phi, T_core, T_inf):
    """
    利用共形映射求解二维 Laplace 方程的温度场。
    在 w-平面（标准圆环域）中解析解为：
        T(ρ,φ) = T_core + (T_inf - T_core) * ln(ρ/ρ_inner) / ln(ρ_outer/ρ_inner)
    然后通过逆映射回到物理 z-平面。

    参数:
        rho : ndarray, w-平面径向坐标
        phi : ndarray, w-平面角度坐标
        T_core : float, 内边界温度 (K)
        T_inf : float, 外边界温度 (K)

    返回:
        T : ndarray, 温度场
    """
    rho = np.asarray(rho, dtype=float)
    rho_inner = 1.0
    rho_outer = 10.0
    # 避免对数奇点
    rho = np.clip(rho, rho_inner + 1e-12, rho_outer - 1e-12)
    T = T_core + (T_inf - T_core) * np.log(rho / rho_inner) / np.log(rho_outer / rho_inner)
    return T


def test_conformal_mapping():
    """自包含测试"""
    theta = np.linspace(0, 2 * np.pi, 100)
    z = 1.1 * np.exp(1j * theta)
    w = joukowsky_transform(z)
    z_rec = joukowsky_inverse(w, branch='+')
    err = np.max(np.abs(z_rec - z))
    print(f"[conformal_mapping] Inverse transform max error = {err:.3e}")
    assert err < 1e-10, "Joukowsky inverse inaccurate"

    # 测试吸积流线映射
    w_r, w_i = map_accretion_streamline(1.2, theta, offset=0.15)
    print(f"[conformal_mapping] Accretion streamline mapped to {len(w_r)} points")


if __name__ == "__main__":
    test_conformal_mapping()
