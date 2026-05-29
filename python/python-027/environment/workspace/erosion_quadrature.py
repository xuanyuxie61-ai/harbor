# -*- coding: utf-8 -*-
"""
erosion_quadrature.py
壁材料侵蚀率数值积分模块
基于种子项目 1304_triangle_felippa_rule (三角形积分) 和 144_cc_project (Clenshaw-Curtis/Chebyshev求积) 重构

本模块计算等离子体离子对壁材料的物理溅射产额和能量沉积分布，
使用多种高阶数值积分规则对靶板表面的复杂积分进行精确求解。
"""

import numpy as np
from parameters import get_parameters


class ErosionQuadrature:
    """
    壁材料侵蚀率数值积分器
    
    核心物理量:
        (1) 物理溅射产额 Y(E, theta):
            Y(E) = Q * s_n(E) / (1 + lambda * U_0 / E)
        
        (2) 靶板表面总侵蚀通量:
            Phi_erosion = integral_{surface} Y(E(x), theta(x)) * Gamma_i(x) dA
    
    其中:
        - E: 入射离子能量 [eV]
        - theta: 入射角 [rad]
        - U_0: 表面结合能 [eV]
        - Gamma_i: 离子通量 [m^-2 s^-1]
        - s_n: 核阻止截面（KrC 或 ZBL 势）
    """

    def __init__(self, params=None):
        if params is None:
            params = get_parameters()
        self.params = params
        self.U_0 = params.get('E_bind')
        self.E_th = params.get('E_threshold')
        self.Z_wall = params.get('wall_Z')
        self.M_wall = params.get('wall_M')
        self.Z_i = params.get('Z_i')
        self.M_i = params.get('m_i')

    def nuclear_stopping_krc(self, epsilon):
        """
        KrC (Moliere) 核阻止截面（无量纲化）
        
        公式:
            s_n(epsilon) = 0.5 * ln(1 + 1.2288*epsilon) / (epsilon + 0.1728*sqrt(epsilon) + 0.008*epsilon^0.1504)
        
        其中 epsilon 为无量纲能量（Lindhard 约化能量）。
        
        Parameters:
            epsilon: 约化能量
        
        Returns:
            s_n: 无量纲核阻止截面
        """
        epsilon = np.asarray(epsilon, dtype=float)
        # 避免除零
        eps_safe = np.where(epsilon < 1.0e-10, 1.0e-10, epsilon)

        numerator = 0.5 * np.log(1.0 + 1.2288 * eps_safe)
        denominator = eps_safe + 0.1728 * np.sqrt(eps_safe) + 0.008 * eps_safe**0.1504

        s_n = numerator / denominator
        return s_n

    def reduced_energy(self, E_ev):
        """
        计算 Lindhard 约化能量
        
        公式（SI单位）:
            epsilon = (M2/(M1+M2)) * E[J] * a / (Z1*Z2*e^2/(4*pi*epsilon_0))
        
        其中 a 为屏蔽长度（Bohr 屏蔽）:
            a = 0.8854 * a_0 / sqrt(Z_1^{2/3} + Z_2^{2/3})
        """
        a0 = 0.5291772108e-10  # Bohr半径 [m]
        e_charge = 1.602176634e-19  # C
        epsilon_0 = 8.854187817e-12  # F/m

        # 屏蔽长度
        z_sum = self.Z_i**(2.0/3.0) + self.Z_wall**(2.0/3.0)
        a_screen = 0.8854 * a0 / np.sqrt(z_sum)

        # Coulomb常数因子: Z1*Z2*e^2/(4*pi*epsilon_0)  [J*m]
        coulomb_factor = self.Z_i * self.Z_wall * e_charge**2 / (4.0 * np.pi * epsilon_0)

        # 约化能量（Lindhard单位，无量纲）
        mass_ratio = self.M_wall / (self.M_i + self.M_wall)
        E_joule = E_ev * e_charge
        epsilon = mass_ratio * E_joule * a_screen / coulomb_factor

        # 数值稳定性
        if epsilon < 1.0e-10:
            epsilon = 1.0e-10
        return epsilon

    def sputtering_yield_bohdansky(self, E_ev, theta=0.0):
        """
        Bohdansky 物理溅射产额公式
        
        公式:
            Y(E) = 0.042 * Q * s_n(epsilon) * (E_th/E)^{0.25} * (1 - E_th/E)^{2.5}
        
        其中:
            Q = (Z_wall / Z_i)^{0.2} * (M_i / M_wall)^{0.15}
        
        Parameters:
            E_ev:   入射能量 [eV]
            theta:  入射角 [rad]（0 = 垂直入射）
        
        Returns:
            Y: 溅射产额 [原子/离子]
        """
        E_ev = float(E_ev)
        if E_ev <= self.E_th:
            return 0.0

        epsilon = self.reduced_energy(E_ev)
        s_n = self.nuclear_stopping_krc(epsilon)

        Q = (self.Z_wall / max(self.Z_i, 1))**0.2 * (self.M_i / max(self.M_wall, 1))**0.15

        ratio = self.E_th / E_ev
        if ratio >= 1.0:
            return 0.0

        Y = 0.042 * Q * s_n * (ratio**0.25) * ((1.0 - ratio)**2.5)

        # 角度修正（Yamamura 公式近似）
        if theta != 0.0:
            theta_r = np.radians(78.0)  # 最佳溅射角近似
            cos_theta = np.cos(theta)
            f_theta = np.exp(-((theta - theta_r) / 0.8)**2) + 0.5 * cos_theta**(-1.5)
            f_theta = min(f_theta, 5.0)  # 限制角度增强
            Y *= f_theta

        # 物理约束
        if Y < 0:
            Y = 0.0
        if Y > 100.0:
            Y = 100.0

        return Y

    def energy_deposition_profile(self, E_ev, x_depth):
        """
        计算离子在材料中的能量沉积深度分布
        
        使用高斯近似:
            dE/dx ~ E * exp(-(x - R_p)^2 / (2*sigma^2))
        
        其中 R_p 为投影射程。
        
        Parameters:
            E_ev:    入射能量 [eV]
            x_depth: 深度数组 [m]
        
        Returns:
            dep: 能量沉积分布 [eV/m]
        """
        # 投影射程近似（SRIM简化模型）:
        # R_p [nm] ~ 10 * E_ev^{0.6} / (Z_i * Z_wall * M_i)
        R_p = 1.0e-9 * 10.0 * (E_ev**0.6) / (self.Z_i * self.Z_wall * max(self.M_i, 1))
        sigma = R_p * 0.3

        if sigma <= 0:
            return np.zeros_like(x_depth)

        dep = E_ev * np.exp(-0.5 * ((x_depth - R_p) / sigma)**2) / (sigma * np.sqrt(2.0 * np.pi))
        return dep

    # ---- 数值积分规则 ----

    @staticmethod
    def clenshaw_curtis_rule(n):
        """
        标准 Clenshaw-Curtis 求积规则（基于 cc_standard.m）
        
        积分区间: [-1, 1]
        节点: x_i = cos(pi * (i-1) / (n-1))
        权重: 通过余弦级数计算
        
        Returns:
            x: 节点
            w: 权重
        """
        if n < 1:
            raise ValueError("n 必须 >= 1")

        x = np.zeros(n)
        w = np.zeros(n)

        if n == 1:
            x[0] = 0.0
            w[0] = 2.0
            return x, w

        for i in range(n):
            x[i] = np.cos(np.pi * (n - 1 - i) / (n - 1))

        w[:] = 1.0
        for i in range(n):
            theta = np.pi * i / (n - 1)
            jhi = (n - 1) // 2
            for j in range(1, jhi + 1):
                if 2 * j == n - 1:
                    b = 1.0
                else:
                    b = 2.0
                w[i] -= b * np.cos(2.0 * j * theta) / (4.0 * j * j - 1.0)

        w[0] /= (n - 1)
        w[1:-1] = 2.0 * w[1:-1] / (n - 1)
        w[-1] /= (n - 1)

        return x, w

    @staticmethod
    def chebyshev1_rule(n):
        """
        Gauss-Chebyshev Type 1 求积规则（基于 chebyshev1_compute.m）
        
        积分: integral_{-1}^{1} f(x) / sqrt(1-x^2) dx
        节点: x_i = cos(pi * (2i-1) / (2n))
        权重: w_i = pi / n
        
        Returns:
            x: 节点
            w: 权重
        """
        if n < 1:
            raise ValueError("n 必须 >= 1")

        x = np.zeros(n)
        w = np.full(n, np.pi / n)

        for i in range(n):
            x[i] = np.cos(np.pi * (2.0 * n - 1.0 - 2.0 * i) / (2.0 * n))

        return x, w

    @staticmethod
    def triangle_unit_o03():
        """
        3点三角形求积规则（基于 triangle_unit_o03.m）
        
        积分区域: 单位三角形 0<=x, 0<=y, x+y<=1
        精度: 2次多项式精确
        
        Returns:
            w: 权重 (3,)
            xy: 节点 (2, 3)
        """
        w = np.ones(3) / 3.0
        xy = np.array([
            [2.0/3.0, 1.0/6.0],
            [1.0/6.0, 2.0/3.0],
            [1.0/6.0, 1.0/6.0]
        ]).T
        return w, xy

    @staticmethod
    def triangle_unit_o12():
        """
        12点三角形求积规则（基于 triangle_unit_o12.m）
        
        精度: 6次多项式精确
        
        Returns:
            w: 权重 (12,)
            xy: 节点 (2, 12)
        """
        w = np.array([
            0.050844906370206816921,
            0.050844906370206816921,
            0.050844906370206816921,
            0.11678627572637936603,
            0.11678627572637936603,
            0.11678627572637936603,
            0.082851075618373575194,
            0.082851075618373575194,
            0.082851075618373575194,
            0.082851075618373575194,
            0.082851075618373575194,
            0.082851075618373575194,
        ])
        xy = np.array([
            [0.87382197101699554332, 0.063089014491502228340],
            [0.063089014491502228340, 0.87382197101699554332],
            [0.063089014491502228340, 0.063089014491502228340],
            [0.50142650965817915742, 0.24928674517091042129],
            [0.24928674517091042129, 0.50142650965817915742],
            [0.24928674517091042129, 0.24928674517091042129],
            [0.053145049844816947353, 0.31035245103378440542],
            [0.31035245103378440542, 0.053145049844816947353],
            [0.053145049844816947353, 0.63650249912139864723],
            [0.31035245103378440542, 0.63650249912139864723],
            [0.63650249912139864723, 0.053145049844816947353],
            [0.63650249912139864723, 0.31035245103378440542],
        ]).T
        return w, xy

    @staticmethod
    def triangle_unit_monomial_integral(expon):
        """
        单位三角形上单项式积分（基于 triangle_unit_monomial_integral.m）
        
        integral_{unit triangle} x^m * y^n dx dy = m! * n! / (m + n + 2)!
        
        Parameters:
            expon: [m, n] 指数
        
        Returns:
            value: 积分值
        """
        m, n = int(expon[0]), int(expon[1])
        if m < 0 or n < 0:
            return 0.0

        value = 1.0
        k = m
        for i in range(1, n + 1):
            k += 1
            value *= i / k
        k += 1
        value /= k
        k += 1
        value /= k
        return value

    def integrate_sputtering_yield_1d(self, E_min, E_max, n_points=64):
        """
        使用 Clenshaw-Curtis 规则积分溅射产额随能量的分布
        
        integral_{E_min}^{E_max} Y(E) * f(E) dE
        
        其中 f(E) 假设为 Maxwellian-like 分布:
            f(E) ~ E * exp(-E/T_eff)
        """
        x_cc, w_cc = self.clenshaw_curtis_rule(n_points)

        # 将 [-1, 1] 映射到 [E_min, E_max]
        E_nodes = 0.5 * (E_max - E_min) * x_cc + 0.5 * (E_max + E_min)
        jacobian = 0.5 * (E_max - E_min)

        # Maxwellian-like 权重
        T_eff = self.params.get('T_e')
        if T_eff <= 0:
            T_eff = 50.0

        integrand = np.zeros(n_points)
        for i in range(n_points):
            E = E_nodes[i]
            if E <= self.E_th:
                integrand[i] = 0.0
            else:
                Y = self.sputtering_yield_bohdansky(E)
                f_E = (E / T_eff) * np.exp(-E / T_eff)
                integrand[i] = Y * f_E

        result = jacobian * np.sum(w_cc * integrand)
        return result, E_nodes, integrand

    def integrate_erosion_over_triangle(self, triangle_vertices, gamma_func, E_func):
        """
        使用三角形高阶求积计算三角形单元上的总侵蚀率
        
        integral_{triangle} Y(E(x,y)) * Gamma_i(x,y) dA
        
        Parameters:
            triangle_vertices: (3, 2) 三角形顶点
            gamma_func: 离子通量函数 Gamma_i(x, y)
            E_func: 离子能量函数 E(x, y)
        
        Returns:
            total_yield: 三角形单元上的总侵蚀产额
        """
        w, xy_ref = self.triangle_unit_o12()

        # 参考坐标到物理坐标的映射
        v0 = triangle_vertices[0]
        v1 = triangle_vertices[1]
        v2 = triangle_vertices[2]

        # Jacobian 行列式 = 2 * area
        J = np.array([[v1[0]-v0[0], v2[0]-v0[0]],
                      [v1[1]-v0[1], v2[1]-v0[1]]])
        det_J = abs(np.linalg.det(J))
        if det_J < 1.0e-20:
            return 0.0

        area = 0.5 * det_J

        total = 0.0
        for i in range(len(w)):
            # 参考坐标
            xi, eta = xy_ref[0, i], xy_ref[1, i]
            # 物理坐标（仿射变换）
            x = v0[0] + xi * (v1[0] - v0[0]) + eta * (v2[0] - v0[0])
            y = v0[1] + xi * (v1[1] - v0[1]) + eta * (v2[1] - v0[1])

            gamma = gamma_func(x, y)
            E = E_func(x, y)

            if E > self.E_th:
                Y = self.sputtering_yield_bohdansky(E)
                total += w[i] * Y * gamma

        # 单位三角形面积 = 1/2，因此结果需要乘以实际面积 / 0.5 = 2*area
        total *= area
        return total


def demo_erosion():
    """演示侵蚀率计算"""
    eq = ErosionQuadrature()

    # 测试溅射产额
    energies = [50, 100, 200, 500, 1000, 2000]
    print("物理溅射产额 (D -> W):")
    for E in energies:
        Y = eq.sputtering_yield_bohdansky(E)
        print(f"  E = {E:5d} eV, Y = {Y:.4f}")

    # 测试积分
    result, _, _ = eq.integrate_sputtering_yield_1d(10.0, 5000.0, n_points=64)
    print(f"\n能量加权平均溅射产额 = {result:.4f}")

    # 测试三角形积分
    tri = np.array([[0.0, 0.0], [1.0e-3, 0.0], [0.5e-3, 1.0e-3]])
    def gamma_f(x, y):
        return 1.0e22
    def E_f(x, y):
        return 500.0
    erosion = eq.integrate_erosion_over_triangle(tri, gamma_f, E_f)
    print(f"三角形单元侵蚀率 = {erosion:.3e} 原子/s")

    return eq


if __name__ == "__main__":
    demo_erosion()
