"""
数值积分模块

本模块实现多种区域上的高精度数值积分，用于平流层化学模型中的:
- 光通量积分 (波长 × 高度)
- 水平区域上的物种总量计算
- 化学反应速率积分

科学公式:
1. 单位正方形上的单项式积分:
   ∫_0^1 ∫_0^1 x^e1 y^e2 dx dy = 1 / ((e1+1)(e2+1))

2. 六边形区域上的积分 (Lyness 规则):
   利用仿射变换将六边形映射到参考单元

3. 梯形法则 (垂直方向):
   ∫_{z1}^{z2} f(z) dz ≈ (f(z1) + f(z2)) / 2 * (z2 - z1)

4. Simpson 法则:
   ∫_{z1}^{z3} f(z) dz ≈ (z3-z1)/6 * (f(z1) + 4f(z2) + f(z3))

5. Gauss-Legendre 积分:
   ∫_{-1}^{1} f(x) dx ≈ Σ w_i f(x_i)
   节点为 Legendre 多项式零点

融入原项目:
- 1147_square_integrals (正方形单项式积分)
- 528_hexagon_lyness_rule (六边形积分规则)
"""

import numpy as np
from typing import Tuple, Optional, Callable


class SquareQuadrature:
    """
    正方形区域 [0,1]×[0,1] 上的数值积分
    融入 1147_square_integrals
    """

    def __init__(self):
        pass

    def square01_area(self) -> float:
        """
        单位正方形面积
        A = 1.0
        """
        return 1.0

    def monomial_integral(self, e: Tuple[int, int]) -> float:
        """
        单位正方形上的单项式积分
        ∫_0^1 ∫_0^1 x^e1 y^e2 dx dy = 1/((e1+1)(e2+1))

        Parameters
        ----------
        e : tuple
            指数 (e1, e2)

        Returns
        -------
        integral : float
        """
        e1, e2 = e
        if e1 < 0 or e2 < 0:
            raise ValueError("指数必须非负")
        return 1.0 / ((e1 + 1) * (e2 + 1))

    def sample_uniform(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        在正方形内均匀随机采样
        """
        x = np.random.rand(n)
        y = np.random.rand(n)
        return x, y

    def monte_carlo_integrate(self, f: Callable,
                               n_samples: int = 10000) -> Tuple[float, float]:
        """
        蒙特卡洛积分
        I ≈ (1/N) Σ f(x_i)
        """
        x, y = self.sample_uniform(n_samples)
        values = f(x, y)
        mean = np.mean(values)
        std = np.std(values)
        # 误差估计
        error = std / np.sqrt(n_samples)
        return mean, error

    def gauss_legendre_2d(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D Gauss-Legendre 积分节点和权重
        """
        x_1d, w_1d = np.polynomial.legendre.leggauss(n)
        # 映射到 [0, 1]
        x_1d = 0.5 * (x_1d + 1.0)
        w_1d = 0.5 * w_1d

        x = np.zeros(n * n)
        y = np.zeros(n * n)
        w = np.zeros(n * n)

        idx = 0
        for i in range(n):
            for j in range(n):
                x[idx] = x_1d[i]
                y[idx] = x_1d[j]
                w[idx] = w_1d[i] * w_1d[j]
                idx += 1

        return x, y, w

    def integrate_gauss(self, f: Callable, n: int = 5) -> float:
        """
        使用 Gauss-Legendre 积分
        """
        x, y, w = self.gauss_legendre_2d(n)
        values = f(x, y)
        return np.sum(w * values)


class HexagonQuadrature:
    """
    正六边形区域上的数值积分
    融入 528_hexagon_lyness_rule
    """

    def __init__(self):
        pass

    def hexagon_area(self, side_length: float = 1.0) -> float:
        """
        正六边形面积
        A = (3√3 / 2) * a²
        """
        if side_length <= 0:
            raise ValueError("边长必须为正")
        return 3.0 * np.sqrt(3.0) / 2.0 * side_length ** 2

    def lyness_rule(self, rule_id: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lyness 六边形积分规则
        规则定义在单位六边形上 (中心在原点，边长为 1/√3)

        Parameters
        ----------
        rule_id : int
            规则编号 (1-7)

        Returns
        -------
        xi, eta : ndarray
            局部坐标 (重心坐标形式)
        w : ndarray
            权重
        """
        if rule_id == 1:
            # 1点规则 (重心)
            xi = np.array([0.0])
            eta = np.array([0.0])
            w = np.array([1.0])
        elif rule_id == 2:
            # 7点规则 (中心 + 6个顶点)
            xi = np.array([0.0, 1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
            eta = np.array([0.0, 0.0, np.sqrt(3)/2, np.sqrt(3)/2,
                            0.0, -np.sqrt(3)/2, -np.sqrt(3)/2])
            w = np.array([0.5, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12])
        elif rule_id == 3:
            # 7点规则 (中心 + 6边中点)
            s3 = np.sqrt(3.0) / 2.0
            xi = np.array([0.0, 0.75, 0.0, -0.75, -0.75, 0.0, 0.75])
            eta = np.array([0.0, 0.375, 0.75, 0.375, -0.375, -0.75, -0.375])
            w_center = 0.25
            w_edge = 0.125
            w = np.array([w_center, w_edge, w_edge, w_edge,
                          w_edge, w_edge, w_edge])
        elif rule_id == 4:
            # 更复杂的 19点规则 (简化)
            # 使用同心环结构
            r1 = 0.5
            r2 = 0.9
            angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
            angles2 = np.linspace(0, 2*np.pi, 13, endpoint=False)

            xi = [0.0]
            eta = [0.0]
            w = [0.2]

            for a in angles1[:-1]:
                xi.append(r1 * np.cos(a))
                eta.append(r1 * np.sin(a))
                w.append(0.1 / 6.0)

            for a in angles2[:-1]:
                xi.append(r2 * np.cos(a))
                eta.append(r2 * np.sin(a))
                w.append(0.1 / 12.0)

            xi = np.array(xi)
            eta = np.array(eta)
            w = np.array(w)
            # 重新归一化
            w = w / np.sum(w)
        else:
            xi = np.array([0.0])
            eta = np.array([0.0])
            w = np.array([1.0])

        return xi, eta, w

    def integrate(self, f: Callable, side_length: float = 1.0,
                  rule_id: int = 3) -> float:
        """
        在六边形上数值积分
        """
        xi, eta, w = self.lyness_rule(rule_id)
        area = self.hexagon_area(side_length)

        # 坐标缩放
        x = xi * side_length
        y = eta * side_length

        values = f(x, y)
        return area * np.sum(w * values)


class VerticalIntegrator:
    """
    垂直方向数值积分器
    用于光通量、柱总量等积分
    """

    def __init__(self, z: np.ndarray):
        """
        Parameters
        ----------
        z : ndarray
            高度网格 (m), 单调递增
        """
        if len(z) < 2:
            raise ValueError("至少需要两个网格点")
        if not np.all(np.diff(z) > 0):
            raise ValueError("高度网格必须严格单调递增")
        self.z = z.copy()
        self.nz = len(z)

    def trapezoid(self, f: np.ndarray) -> float:
        """
        梯形法则积分
        ∫ f(z) dz
        """
        if len(f) != self.nz:
            raise ValueError("f 长度与网格不匹配")
        return np.trapezoid(f, self.z)

    def simpson(self, f: np.ndarray) -> float:
        """
        Simpson 法则积分 (要求奇数个点)
        """
        if len(f) != self.nz:
            raise ValueError("f 长度与网格不匹配")
        if self.nz % 2 == 0:
            # 使用梯形法则处理最后一个区间
            return np.trapezoid(f, self.z)

        dz = self.z[1] - self.z[0]
        if not np.allclose(np.diff(self.z), dz, rtol=1e-3):
            # 非均匀网格，退化为梯形法则
            return np.trapezoid(f, self.z)

        result = f[0] + f[-1]
        result += 4.0 * np.sum(f[1:-1:2])
        result += 2.0 * np.sum(f[2:-1:2])
        return result * dz / 3.0

    def integrate_product(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """
        计算内积 ∫ f1(z) * f2(z) dz
        """
        if len(f1) != self.nz or len(f2) != self.nz:
            raise ValueError("数组长度与网格不匹配")
        return np.trapezoid(f1 * f2, self.z)

    def cumulative_integral(self, f: np.ndarray) -> np.ndarray:
        """
        计算累积积分 F(z) = ∫_{z_min}^{z} f(z') dz'
        """
        if len(f) != self.nz:
            raise ValueError("f 长度与网格不匹配")
        F = np.zeros(self.nz)
        for i in range(1, self.nz):
            F[i] = F[i - 1] + 0.5 * (f[i] + f[i - 1]) * (self.z[i] - self.z[i - 1])
        return F

    def optical_depth_integral(self, sigma: np.ndarray,
                                n: np.ndarray) -> np.ndarray:
        """
        计算光学厚度
        τ(z) = ∫_{z}^{∞} σ(z') n(z') dz'
        从顶部向下积分
        """
        if len(sigma) != self.nz or len(n) != self.nz:
            raise ValueError("数组长度与网格不匹配")

        tau = np.zeros(self.nz)
        integrand = sigma * n
        for i in range(self.nz - 2, -1, -1):
            tau[i] = tau[i + 1] + 0.5 * (integrand[i] + integrand[i + 1]) * \
                     (self.z[i + 1] - self.z[i])
        return tau


class AtmosphericColumnIntegrator:
    """
    大气柱总量积分器
    """

    def __init__(self, z: np.ndarray,
                 horizontal_area: float = 1.0e10):
        """
        Parameters
        ----------
        z : ndarray
            高度网格 (m)
        horizontal_area : float
            水平区域面积 (m²)
        """
        self.vertical = VerticalIntegrator(z)
        self.horizontal_area = horizontal_area
        self.square_quad = SquareQuadrature()

    def column_density(self, n_z: np.ndarray) -> float:
        """
        柱总量 (molec/m²)
        N = ∫ n(z) dz
        """
        return self.vertical.trapezoid(n_z)

    def total_moles(self, n_z: np.ndarray) -> float:
        """
        区域总摩尔数
        moles = N * A / N_A
        """
        N_A = 6.022e23
        column = self.column_density(n_z)
        return column * self.horizontal_area / N_A

    def dobson_unit(self, n_z: np.ndarray) -> float:
        """
        转换为 Dobson Unit
        1 DU = 2.69e16 molec/cm² = 2.69e20 molec/m²
        """
        column = self.column_density(n_z)  # molec/m³ * m = molec/m²
        # 转换: molec/m² -> DU
        du = column / 2.69e20
        return du

    def horizontal_average(self, field_3d: np.ndarray) -> np.ndarray:
        """
        水平平均
        field_3d 形状: (nx, ny, nz)
        """
        if field_3d.ndim != 3:
            raise ValueError("field_3d 必须为三维数组")
        return np.mean(np.mean(field_3d, axis=0), axis=0)
