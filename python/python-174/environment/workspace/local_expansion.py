"""
local_expansion.py
局部展开模块

融合种子项目:
- 520_hermite_integrands (Hermite高斯积分)
- 1409_wedge_integrals (楔形体积分)
- 777_monomial_value (单项式求值)

科学背景:
局部展开用于描述远场源对局部区域内场点的势能贡献。
在FMM中, 远场多极展开被转换为局部展开 (M2L转换),
然后在局部区域内高效评估。

核心公式:
    对于位于 x_j (|x_j| > R) 的源, 对区域 |x| < R 内的点展开:

    Phi(x) = sum_j q_j / |x - x_j|
           = sum_j q_j * sum_{l=0}^{inf} sum_{m=-l}^{l} (r^l / r_j^{l+1}) * Y_l^m(theta, phi) * conj(Y_l^m(theta_j, phi_j))
           = sum_{l=0}^{inf} sum_{m=-l}^{l} L_l^m * r^l * Y_l^m(theta, phi)

    局部展开系数:
        L_l^m = sum_j q_j / r_j^{l+1} * conj(Y_l^m(theta_j, phi_j))

    实数形式:
        L_l^{m,c} = sum_j q_j / r_j^{l+1} * Y_l^{m,c}(theta_j, phi_j)
        L_l^{m,s} = sum_j q_j / r_j^{l+1} * Y_l^{m,s}(theta_j, phi_j)

    局部区域内势能:
        Phi(x) = sum_{l=0}^{L} sum_{m=0}^{l} r^l * [
                     L_l^{m,c} * Y_l^{m,c}(theta, phi) +
                     L_l^{m,s} * Y_l^{m,s}(theta, phi)
                 ]

Hermite高斯积分用于计算奇异核在局部区域的积分:
    Integral_{-inf}^{+inf} exp(-x^2) * f(x) dx ≈ sum_i w_i * f(x_i)
    其中 x_i 是 Hermite 多项式 H_n(x) 的零点, w_i 是对应权重

楔形体积分思想:
    若源分布在楔形体 W = {(x,y,z): x>=0, y>=0, x+y<=1, -1<=z<=1} 上:
        L_l^m = integral_W rho(x) / |x - c|^{l+1} * Y_l^m(theta,phi) dV
    其中体积元 dV = dx dy dz
"""

import numpy as np
from spherical_geometry import legendre_associated_normalized


def hermite_gauss_nodes_weights(n):
    """
    计算n点Hermite-Gauss积分的节点和权重
    (融合520_hermite_integrands)

    公式:
        节点 x_i: H_n(x_i) = 0 的根
        权重 w_i = 2^{n-1} * n! * sqrt(pi) / (n^2 * [H_{n-1}(x_i)]^2)
        
        Integral exp(-x^2) f(x) dx ≈ sum_i w_i * f(x_i)
    
    参数:
        n: int, 积分点数
    
    返回:
        x, w: ndarray (n,), 节点和权重
    """
    if n <= 0:
        raise ValueError("n必须为正整数")
    # 使用numpy的hermite函数
    # H_n(x) = (-1)^n * exp(x^2) * d^n/dx^n exp(-x^2)
    # 使用SciPy风格的多项式根
    # 这里用numpy的hermite polynomial近似
    # numpy.polynomial.hermite.hermgauss
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)
    return x, w


def integrate_kernel_hermite(f, n=16):
    """
    使用Hermite-Gauss积分计算 Integral_{-inf}^{+inf} K(x) dx
    
    若核函数可写为 exp(-x^2) * g(x) 形式, 则:
        Integral K(x) dx = sum_i w_i * g(x_i)
    
    参数:
        f: callable, 函数 g(x) (即 K(x)/exp(-x^2))
        n: int, 积分点数
    
    返回:
        float, 积分值
    """
    x, w = hermite_gauss_nodes_weights(n)
    vals = np.array([f(xi) for xi in x])
    # numpy的hermgauss权重已经包含 exp(x_i^2) 因子
    # 实际上 hermgauss 计算的是 Integral exp(-x^2) f(x) dx ≈ sum w_i f(x_i)
    # 所以直接 sum w_i * g(x_i) 即可
    return float(np.sum(w * vals))


class LocalExpansion:
    """
    局部展开类
    
    管理局部展开系数, 支持区域内势能/力计算
    """

    def __init__(self, center, order):
        """
        参数:
            center: ndarray (3,), 展开中心
            order: int, 截断阶数 L
        """
        self.center = np.asarray(center, dtype=float)
        self.order = int(order)
        if self.order < 0:
            raise ValueError("order必须非负")
        self.coeffs_real = []
        self.coeffs_imag = []
        for l in range(self.order + 1):
            self.coeffs_real.append(np.zeros(l + 1))
            self.coeffs_imag.append(np.zeros(l + 1))

    def add_source_contribution(self, source_points, source_charges):
        """
        直接累加远场源的局部展开系数
        
        公式:
            L_l^{m,c} += sum_j q_j / r_j^{l+1} * Y_l^{m,c}(theta_j, phi_j)
            L_l^{m,s} += sum_j q_j / r_j^{l+1} * Y_l^{m,s}(theta_j, phi_j)
        
        参数:
            source_points: ndarray (N, 3)
            source_charges: ndarray (N,)
        """
        source_points = np.atleast_2d(source_points)
        source_charges = np.asarray(source_charges, dtype=float)
        if source_points.shape[0] != source_charges.shape[0]:
            raise ValueError("长度不匹配")

        local = source_points - self.center
        r = np.linalg.norm(local, axis=1)
        r = np.where(r < 1e-15, 1e-15, r)
        theta = np.arccos(np.clip(local[:, 2] / r, -1.0, 1.0))
        phi = np.arctan2(local[:, 1], local[:, 0])
        phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)

        for i in range(source_points.shape[0]):
            for l in range(self.order + 1):
                inv_r_lp1 = 1.0 / (r[i] ** (l + 1))
                plm_0 = legendre_associated_normalized(l, 0, np.cos(theta[i]))
                y_0 = plm_0[l]
                self.coeffs_real[l][0] += source_charges[i] * inv_r_lp1 * y_0
                for m in range(1, l + 1):
                    plm_m = legendre_associated_normalized(l, m, np.cos(theta[i]))
                    y_real = plm_m[l] * np.cos(m * phi[i])
                    y_imag = plm_m[l] * np.sin(m * phi[i])
                    self.coeffs_real[l][m] += source_charges[i] * inv_r_lp1 * y_real
                    self.coeffs_imag[l][m] += source_charges[i] * inv_r_lp1 * y_imag

    def evaluate_potential(self, target):
        """
        在目标点评估局部展开势能
        
        公式:
            Phi(x) = sum_{l=0}^{L} sum_{m=0}^{l} R^l * [
                         L_l^{m,c} * Y_l^{m,c}(Theta, Phi) +
                         L_l^{m,s} * Y_l^{m,s}(Theta, Phi)
                     ]
        
        参数:
            target: ndarray (N, 3)
        
        返回:
            ndarray (N,)
        """
        target = np.atleast_2d(target)
        local = target - self.center
        R = np.linalg.norm(local, axis=1)
        R = np.where(R < 1e-15, 1e-15, R)
        theta = np.arccos(np.clip(local[:, 2] / R, -1.0, 1.0))
        phi = np.arctan2(local[:, 1], local[:, 0])
        phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)

        N = target.shape[0]
        potential = np.zeros(N)

        for l in range(self.order + 1):
            R_l = R ** l
            plm_0 = np.array([legendre_associated_normalized(l, 0, np.cos(t)) for t in theta])
            y_0 = plm_0[:, l]
            potential += self.coeffs_real[l][0] * R_l * y_0
            for m in range(1, l + 1):
                plm_m = np.array([legendre_associated_normalized(l, m, np.cos(t)) for t in theta])
                y_real = plm_m[:, l] * np.cos(m * phi)
                y_imag = plm_m[:, l] * np.sin(m * phi)
                potential += R_l * (
                    self.coeffs_real[l][m] * y_real +
                    self.coeffs_imag[l][m] * y_imag
                )
        return potential

    def evaluate_field(self, target):
        """
        数值差分计算电场
        """
        target = np.atleast_2d(target)
        N = target.shape[0]
        h = 1e-6
        field = np.zeros((N, 3))
        for d in range(3):
            offset = np.zeros(3)
            offset[d] = h
            phi_plus = self.evaluate_potential(target + offset)
            phi_minus = self.evaluate_potential(target - offset)
            field[:, d] = -(phi_plus - phi_minus) / (2.0 * h)
        return field

    def wedge_moment_integral(self, exponents):
        """
        计算楔形体上的单项式积分 (融合1409_wedge_integrals思想)
        
        楔形体 W: x>=0, y>=0, x+y<=1, -1<=z<=1
        积分: I = integral_W x^{e1} y^{e2} z^{e3} dV
        
        公式:
            I_xy = integral_{x>=0,y>=0,x+y<=1} x^{e1} y^{e2} dx dy
                 = e2! / [(e1+e2+1)! / e1!] 的变体
            I_z = integral_{-1}^{1} z^{e3} dz
                 = 0           (e3为奇数)
                 = 2/(e3+1)    (e3为偶数)
            I = I_xy * I_z
        
        参数:
            exponents: iterable (e1, e2, e3)
        
        返回:
            float, 积分值
        """
        e1, e2, e3 = int(exponents[0]), int(exponents[1]), int(exponents[2])
        if e1 < 0 or e2 < 0:
            raise ValueError("e1,e2必须非负")
        if e3 == -1:
            raise ValueError("e3不能为-1")

        # I_xy 计算
        value = 1.0
        k = e1
        for i in range(1, e2 + 1):
            k = k + 1
            value = value * i / k
        k = k + 1
        value = value / k
        k = k + 1
        value = value / k

        # I_z
        if e3 % 2 == 1:
            value = 0.0
        else:
            value = value * 2.0 / (e3 + 1)

        return float(value)

    def get_coefficients_norm(self):
        """计算局部展开系数的Frobenius范数"""
        norm_sq = 0.0
        for l in range(self.order + 1):
            for m in range(l + 1):
                norm_sq += self.coeffs_real[l][m] ** 2 + self.coeffs_imag[l][m] ** 2
        return np.sqrt(norm_sq)
