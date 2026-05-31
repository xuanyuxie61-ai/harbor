# -*- coding: utf-8 -*-
"""
geometry_pasta.py
核pasta相几何结构与多面体积分

本模块实现中子星crust层核pasta相的各种几何结构（球状gnocchi、
柱状spaghetti、片状lasagna、管状anti-spaghetti、泡状anti-gnocchi）
的体积、表面积、形状因子计算，并融入多种积分规则:

- 四面体Felippa积分规则 (1246_tetrahedron_felippa_rule)
- 楔形体积分 (1409_wedge_integrals)
- 单位三角形积分 (1326_triangle01_integrals)
- 六边形Stroud积分 (530_hexagon_stroud_rule)
- 四边形网格插值 (956_quadrilateral_surface_display, 去除可视化)

核心物理公式:
1. Wigner-Seitz单元体积:
   V_WS = 1 / n_d    (n_d为核子数密度)
   
2. 填充率 (filling fraction):
   u = V_nucleus / V_WS
   
3. 球状相半径:
   R = (3 u V_WS / 4pi)^{1/3}
   
4. 柱状相半径:
   R = (u V_WS / pi L)^{1/2}   (L为柱长, 通常取 V_WS^{1/3})
   
5. 片状相厚度:
   t = u V_WS^{1/3}
   
6. 形状因子 (Coulomb修正):
   F_C = (3/5)u^{2/3}  (球状)
   F_C = (1/2)u        (柱状)
   F_C = u^{2/3}       (片状)
   
7. 表面面积/体积比 (S/V):
   球状: S/V = 3/R
   柱状: S/V = 2/R
   片状: S/V = 2/t
"""

import numpy as np

# 物理常数
E_CHARGE = 1.43996448  # e^2 in MeV·fm


# ============== 来自1326_triangle01_integrals ==============
def triangle01_area():
    """单位三角形面积: {(x,y) | x>=0, y>=0, x+y<=1}."""
    return 0.5


def triangle01_monomial_integral(e):
    """
    单位三角形上的单项式积分.
    积分: x^e[0] * y^e[1] dx dy
    公式: e1! e2! / (e1+e2+2)!
    """
    e = np.asarray(e)
    if np.any(e < 0):
        raise ValueError("指数必须非负")
    k = 0
    integral = 1.0
    for i in range(2):
        for j in range(1, e[i] + 1):
            k += 1
            integral = integral * j / k
    for _ in range(2):
        k += 1
        integral = integral / k
    return integral


def triangle01_sample(n):
    """在单位三角形中均匀采样n个点."""
    u = np.random.rand(n)
    v = np.random.rand(n)
    mask = u + v > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    return np.column_stack((u, v))


# ============== 来自1409_wedge_integrals ==============
def wedge01_volume():
    """单位楔形体体积."""
    return 1.0


def wedge01_monomial_integral(e):
    """
    单位楔形体上的单项式积分.
    区域: 0<=x, 0<=y, x+y<=1, -1<=z<=1
    公式: value = (积分xy部分) * (2 / (e3+1)) 若e3为偶数
    """
    e = np.asarray(e)
    if np.any(e[:2] < 0):
        raise ValueError("x,y指数必须非负")
    if e[2] == -1:
        raise ValueError("e[3] = -1非法")

    value = 1.0
    k = e[0]
    for i in range(1, e[1] + 1):
        k += 1
        value = value * i / k
    k += 1
    value = value / k
    k += 1
    value = value / k

    if e[2] % 2 == 1:
        value = 0.0
    else:
        value = value * 2.0 / (e[2] + 1)
    return value


# ============== 来自1246_tetrahedron_felippa_rule ==============
def tetrahedron_unit_volume():
    """单位四面体体积: x>=0, y>=0, z>=0, x+y+z<=1."""
    return 1.0 / 6.0


def tetrahedron_unit_monomial(expon):
    """
    单位四面体上的单项式积分.
    积分: x^l y^m z^n dx dy dz
    公式: l! m! n! / (l+m+n+3)!
    """
    expon = np.asarray(expon)
    if np.any(expon < 0):
        raise ValueError("指数必须非负")

    value = 1.0
    k = expon[0]
    for i in range(1, expon[1] + 1):
        k += 1
        value = value * i / k
    for i in range(1, expon[2] + 1):
        k += 1
        value = value * i / k
    for _ in range(3):
        k += 1
        value = value / k
    return value


# ============== 来自530_hexagon_stroud_rule ==============
def hexagon01_area():
    """单位六边形面积 (内切圆半径为1)."""
    return 2.0 * np.sqrt(3.0)


def hexagon_stroud_rule1():
    """Stroud六边形积分规则#1 (1点, 精度1)."""
    n = 1
    p = 1
    x = np.array([0.0])
    y = np.array([0.0])
    w = np.array([1.0])
    return n, p, x, y, w


def hexagon_stroud_rule2():
    """Stroud六边形积分规则#2 (6点, 精度3)."""
    n = 6
    p = 3
    r = np.sqrt(2.0 / 3.0)
    theta = np.linspace(0, 2 * np.pi, 7)[:-1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    w = np.ones(n) / n
    return n, p, x, y, w


def hexagon_stroud_rule3():
    """Stroud六边形积分规则#3 (7点, 精度5)."""
    n = 7
    p = 5
    x = np.zeros(n)
    y = np.zeros(n)
    w = np.zeros(n)
    # 中心点
    x[0] = 0.0
    y[0] = 0.0
    w[0] = 1.0 / 4.0
    # 6个角点
    r = np.sqrt(6.0 / 7.0)
    theta = np.linspace(0, 2 * np.pi, 7)[:-1]
    x[1:] = r * np.cos(theta)
    y[1:] = r * np.sin(theta)
    w[1:] = 1.0 / 8.0
    return n, p, x, y, w


def hexagon_stroud_rule4():
    """Stroud六边形积分规则#4 (12点, 精度7)."""
    n = 12
    p = 7
    x = np.zeros(n)
    y = np.zeros(n)
    w = np.zeros(n)
    # 两组6点
    r1 = np.sqrt((6.0 - np.sqrt(6.0)) / 10.0)
    r2 = np.sqrt((6.0 + np.sqrt(6.0)) / 10.0)
    theta = np.linspace(0, 2 * np.pi, 7)[:-1]
    x[:6] = r1 * np.cos(theta)
    y[:6] = r1 * np.sin(theta)
    w[:6] = (3.0 + 2.0 * np.sqrt(6.0)) / 72.0
    x[6:] = r2 * np.cos(theta)
    y[6:] = r2 * np.sin(theta)
    w[6:] = (3.0 - 2.0 * np.sqrt(6.0)) / 72.0
    return n, p, x, y, w


def hexagon_integral(func, rule=3):
    """
    使用Stroud规则在六边形上积分.
    
    输入:
        func: 函数句柄 func(x,y) -> scalar or array
        rule: 1-4, 选择积分规则
    """
    rules = {
        1: hexagon_stroud_rule1,
        2: hexagon_stroud_rule2,
        3: hexagon_stroud_rule3,
        4: hexagon_stroud_rule4,
    }
    if rule not in rules:
        raise ValueError("rule必须在1-4之间")
    n, p, x, y, w = rules[rule]()
    area = hexagon01_area()
    result = 0.0
    for i in range(n):
        result += w[i] * func(x[i], y[i])
    return result * area


# ============== 来自956_quadrilateral_surface_display ==============
def quadrilateral_bilinear_interpolate(x, y, z_values, xi, yi):
    """
    四边形上的双线性插值 (去除可视化).
    
    输入:
        x: [x1,x2,x3,x4] 四边形顶点x坐标
        y: [y1,y2,y3,y4] 四边形顶点y坐标
        z_values: [z1,z2,z3,z4] 顶点函数值
        xi, yi: 插值点坐标
    输出:
        zi: 插值结果
    """
    # 转换到参考单元 [-1,1]x[-1,1]
    # 使用近似逆映射 (简化: 假设为矩形)
    x = np.asarray(x)
    y = np.asarray(y)
    z_values = np.asarray(z_values)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    if xmax - xmin < 1e-15 or ymax - ymin < 1e-15:
        return np.mean(z_values)

    s = 2.0 * (xi - xmin) / (xmax - xmin) - 1.0
    t = 2.0 * (yi - ymin) / (ymax - ymin) - 1.0

    # 双线性基函数
    N1 = 0.25 * (1.0 - s) * (1.0 - t)
    N2 = 0.25 * (1.0 + s) * (1.0 - t)
    N3 = 0.25 * (1.0 + s) * (1.0 + t)
    N4 = 0.25 * (1.0 - s) * (1.0 + t)

    zi = N1 * z_values[0] + N2 * z_values[1] + N3 * z_values[2] + N4 * z_values[3]
    return zi


# ============== Pasta相几何定义 ==============
class PastaPhase:
    """核pasta相几何基类."""

    PHASE_NAMES = {
        1: 'gnocchi',
        2: 'spaghetti',
        3: 'lasagna',
        4: 'anti-spaghetti',
        5: 'anti-gnocchi'
    }

    def __init__(self, phase_id, density, proton_fraction, u=None):
        """
        输入:
            phase_id: 1-5
            density: 总核子数密度 (fm^{-3})
            proton_fraction: 质子分数
            u: 填充率 (默认自动计算)
        """
        if phase_id not in self.PHASE_NAMES:
            raise ValueError(f"phase_id必须在1-5之间, 得到{phase_id}")
        if density <= 0.0:
            raise ValueError("密度必须大于0")
        if proton_fraction < 0.0 or proton_fraction > 1.0:
            raise ValueError("质子分数必须在[0,1]之间")

        self.phase_id = phase_id
        self.density = density
        self.proton_fraction = proton_fraction
        self.rho_n = density * (1.0 - proton_fraction)
        self.rho_p = density * proton_fraction

        # Wigner-Seitz单元体积
        self.V_WS = 1.0 / density
        self.a_WS = self.V_WS ** (1.0 / 3.0)

        # 填充率: 默认使用能量最小化近似
        if u is None:
            self.u = self._optimal_filling()
        else:
            if u <= 0.0 or u >= 1.0:
                raise ValueError("填充率必须在(0,1)之间")
            self.u = u

        self._compute_geometry()

    def _optimal_filling(self):
        """近似最优填充率 (简化的Thomas-Fermi近似)."""
        # 典型值在0.2-0.5之间
        return 0.3 + 0.1 * self.proton_fraction

    def _compute_geometry(self):
        """计算几何参数."""
        raise NotImplementedError

    def surface_area(self):
        """返回表面积 (fm^2)."""
        raise NotImplementedError

    def volume(self):
        """返回体积 (fm^3)."""
        raise NotImplementedError

    def coulomb_factor(self):
        """库仑形状因子."""
        raise NotImplementedError

    def surface_to_volume(self):
        """表面积/体积比."""
        return self.surface_area() / self.volume()


class GnocchiPhase(PastaPhase):
    """球状相 (核物质球在核子气体中)."""

    def __init__(self, density, proton_fraction, u=None):
        super().__init__(1, density, proton_fraction, u)

    def _compute_geometry(self):
        self.R = (3.0 * self.u * self.V_WS / (4.0 * np.pi)) ** (1.0 / 3.0)

    def surface_area(self):
        return 4.0 * np.pi * self.R**2

    def volume(self):
        return (4.0 / 3.0) * np.pi * self.R**3

    def coulomb_factor(self):
        return (3.0 / 5.0) * self.u ** (2.0 / 3.0)


class SpaghettiPhase(PastaPhase):
    """柱状相 (核物质柱在核子气体中)."""

    def __init__(self, density, proton_fraction, u=None):
        super().__init__(2, density, proton_fraction, u)

    def _compute_geometry(self):
        # 柱长取 Wigner-Seitz边长
        self.L = self.a_WS
        self.R = np.sqrt(self.u * self.V_WS / (np.pi * self.L))

    def surface_area(self):
        return 2.0 * np.pi * self.R * self.L

    def volume(self):
        return np.pi * self.R**2 * self.L

    def coulomb_factor(self):
        return 0.5 * self.u


class LasagnaPhase(PastaPhase):
    """片状相 (核物质片在核子气体中)."""

    def __init__(self, density, proton_fraction, u=None):
        super().__init__(3, density, proton_fraction, u)

    def _compute_geometry(self):
        self.t = self.u * self.a_WS
        self.A_slice = self.a_WS**2

    def surface_area(self):
        return 2.0 * self.A_slice

    def volume(self):
        return self.t * self.A_slice

    def coulomb_factor(self):
        return self.u ** (2.0 / 3.0)


class AntiSpaghettiPhase(PastaPhase):
    """管状相 (核子气体柱在核物质中)."""

    def __init__(self, density, proton_fraction, u=None):
        super().__init__(4, density, proton_fraction, u)

    def _compute_geometry(self):
        self.L = self.a_WS
        # 气体柱半径: 填充率是气体体积占比
        self.R = np.sqrt((1.0 - self.u) * self.V_WS / (np.pi * self.L))

    def surface_area(self):
        return 2.0 * np.pi * self.R * self.L

    def volume(self):
        # 返回气体柱体积
        return np.pi * self.R**2 * self.L

    def coulomb_factor(self):
        # 反相: 1 - u的某种形式
        return 0.5 * (1.0 - self.u)


class AntiGnocchiPhase(PastaPhase):
    """泡状相 (核子气体泡在核物质中)."""

    def __init__(self, density, proton_fraction, u=None):
        super().__init__(5, density, proton_fraction, u)

    def _compute_geometry(self):
        self.R = (3.0 * (1.0 - self.u) * self.V_WS / (4.0 * np.pi)) ** (1.0 / 3.0)

    def surface_area(self):
        return 4.0 * np.pi * self.R**2

    def volume(self):
        return (4.0 / 3.0) * np.pi * self.R**3

    def coulomb_factor(self):
        return (3.0 / 5.0) * (1.0 - self.u) ** (2.0 / 3.0)


def create_pasta_phase(phase_id, density, proton_fraction, u=None):
    """工厂函数创建pasta相实例."""
    constructors = {
        1: GnocchiPhase,
        2: SpaghettiPhase,
        3: LasagnaPhase,
        4: AntiSpaghettiPhase,
        5: AntiGnocchiPhase,
    }
    return constructors[phase_id](density, proton_fraction, u)


def pasta_energy_landscape(density_range, proton_fraction, n_points=20):
    """
    计算不同pasta相的能量景观.
    
    返回:
        dict: {phase_name: {'u_opt': optimal_u, 'E_min': minimum_energy, ...}}
    """
    results = {}
    u_grid = np.linspace(0.05, 0.95, n_points)

    for pid in range(1, 6):
        name = PastaPhase.PHASE_NAMES[pid]
        energies = []
        for rho in density_range:
            e_vals = []
            for u in u_grid:
                try:
                    phase = create_pasta_phase(pid, rho, proton_fraction, u)
                    # 简化能量 = 表面能 + 库仑能 (系数稍后由完整模型提供)
                    sigma = 1.0  # 表面张力系数 (MeV/fm^2), 占位
                    e_surf = sigma * phase.surface_to_volume()
                    e_coul = 0.5 * E_CHARGE * phase.rho_p**2 * phase.coulomb_factor() * phase.volume()
                    e_total = e_surf + e_coul
                    e_vals.append(e_total)
                except ValueError:
                    e_vals.append(np.inf)
            if len(e_vals) > 0:
                energies.append(np.min(e_vals))
            else:
                energies.append(np.inf)
        results[name] = np.array(energies)

    return results


if __name__ == '__main__':
    # 自测试
    rho = 0.08
    x_p = 0.3
    for pid in range(1, 6):
        p = create_pasta_phase(pid, rho, x_p)
        print(f"{p.PHASE_NAMES[pid]}: R/t={getattr(p, 'R', getattr(p, 't', 'N/A')):.3f} fm, "
              f"S/V={p.surface_to_volume():.3f} fm^-1")
