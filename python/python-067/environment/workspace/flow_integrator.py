# -*- coding: utf-8 -*-
"""
flow_integrator.py
裂隙渗流数值积分与通量计算模块

基于种子项目 463_gegenbauer_rule 的高斯-盖根堡尔数值积分算法，
用于精确计算裂隙段内的流量通量和溶质传输积分。

在裂隙介质渗流中，数值积分应用于：
    1. 裂隙段横截面的流速分布积分
    2. 示踪剂突破曲线的矩计算
    3. 有效弥散系数的体积平均

核心公式：
    高斯-盖根堡尔求积：
        ∫_a^b [(x-a)(b-x)]^α f(x) dx ≈ Σ_{i=1}^n w_i f(x_i)
    
    其中节点 x_i 和权重 w_i 由 Jacobi 矩阵的特征值和特征向量确定。
    
    Jacobi 矩阵 J 的元素（Gegenbauer 情况 α = β）：
        a_i = 0
        b_i = i / sqrt(4(i+α)² - 1),  i ≥ 1
    
    裂隙段流量（速度剖面积分）：
        Q = ∫_0^b v(z) W dz
    
    其中 W 为裂隙宽度，b 为开度，v(z) 为局部流速。
    
    对于层流（Cubic Law）：
        v(z) = -(b²/8μ) (dp/dx) [1 - (2z/b - 1)²]
    
    抛物线速度剖面的平均流速：
        v_avg = -(b²/12μ) (dp/dx)
"""

import numpy as np
from typing import Tuple, List, Callable


class GegenbauerQuadrature:
    """
    高斯-盖根堡尔数值积分器

    基于 IQPACK 算法（Elhay & Kautsky, 1987）计算求积节点和权重。
    适用于带有 (x-a)^α (b-x)^α 权函数的积分。
    """

    def __init__(self, order: int, alpha: float, a: float = -1.0, b: float = 1.0):
        """
        Parameters
        ----------
        order : int
            求积阶数（节点数）
        alpha : float
            权函数指数 α > -1
        a, b : float
            积分区间
        """
        if order < 1:
            raise ValueError("order 必须为正整数")
        if alpha <= -1.0:
            raise ValueError("alpha 必须大于 -1")

        self.order = order
        self.alpha = alpha
        self.a = a
        self.b = b
        self.nodes, self.weights = self._compute_rule()

    def _compute_rule(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算盖根堡尔求积节点和权重

        算法步骤：
            1. 构建 Jacobi 矩阵（对称三对角）
            2. 使用隐式 QL 算法求特征值和特征向量
            3. 缩放至目标区间 [a, b]
        """
        n = self.order
        alpha = self.alpha

        # 构建 Jacobi 矩阵对角线和次对角线
        # Gegenbauer 情况: a_i = 0, b_i = sqrt(i(i+2α-1) / (4(i+α)²-1))
        diag = np.zeros(n)
        offdiag = np.zeros(n - 1)

        for i in range(1, n):
            num = i * (i + 2.0 * alpha)
            den = (2.0 * i + 2.0 * alpha - 1.0) * (2.0 * i + 2.0 * alpha + 1.0)
            if den > 0 and num > 0:
                offdiag[i - 1] = np.sqrt(num / den)

        # 使用 numpy 的 eigh 求解（对称三对角矩阵的特征值问题）
        # 构造完整对称矩阵
        J = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
        eigenvalues, eigenvectors = np.linalg.eigh(J)

        # 节点为特征值
        nodes = eigenvalues

        # 权重：w_i = μ_0 * v_{1,i}²
        # 其中 μ_0 = 2^(2α+1) * Gamma(α+1)² / Gamma(2α+2)
        # 对于标准区间 [-1, 1]
        from math import gamma
        mu0 = (2.0 ** (2.0 * alpha + 1.0)) * (gamma(alpha + 1.0) ** 2) / gamma(2.0 * alpha + 2.0)

        weights = mu0 * (eigenvectors[0, :] ** 2)

        # 缩放至 [a, b]
        mid = (self.a + self.b) / 2.0
        scale = (self.b - self.a) / 2.0
        nodes_scaled = mid + scale * nodes
        weights_scaled = scale * weights

        return nodes_scaled, weights_scaled

    def integrate(self, f: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        数值积分

        Parameters
        ----------
        f : callable
            被积函数

        Returns
        -------
        float
            积分近似值
        """
        fx = f(self.nodes)
        return float(np.dot(self.weights, fx))

    def integrate_parabolic_profile(self, dp_dx: float, b: float,
                                     mu: float = 1.0e-3) -> float:
        """
        积分裂隙层流抛物线速度剖面

        速度分布：
            v(z) = -(b²/8μ) (dp/dx) [1 - (2z/b - 1)²]

        积分结果应为平均流速乘以开度：
            Q/W = ∫_0^b v(z) dz = -(b³/12μ) (dp/dx)

        Parameters
        ----------
        dp_dx : float
            压力梯度 [Pa/m]
        b : float
            裂隙开度 [m]
        mu : float
            动力粘度 [Pa·s]

        Returns
        -------
        float
            单位宽度流量 [m²/s]
        """
        if b <= 0 or mu <= 0:
            raise ValueError("b 和 mu 必须为正")

        def velocity(z):
            zeta = 2.0 * z / b - 1.0
            return -(b ** 2 / (8.0 * mu)) * dp_dx * (1.0 - zeta ** 2)

        return self.integrate(velocity)


class FlowIntegrator:
    """
    裂隙渗流积分计算器

    提供裂隙介质渗流分析中的各类积分计算功能。
    """

    @staticmethod
    def breakthrough_curve_moments(times: np.ndarray,
                                    concentrations: np.ndarray) -> dict:
        """
        计算突破曲线的统计矩

        公式：
            M_0 = ∫ C(t) dt          （零阶矩：总质量）
            M_1 = ∫ t C(t) dt        （一阶矩）
            t_mean = M_1 / M_0       （平均突破时间）
            M_2 = ∫ t² C(t) dt       （二阶矩）
            σ_t² = M_2/M_0 - t_mean² （时间方差）

        Parameters
        ----------
        times : np.ndarray
            时间数组 [s]
        concentrations : np.ndarray
            浓度数组 [kg/m³]

        Returns
        -------
        dict
            统计矩
        """
        if len(times) != len(concentrations):
            raise ValueError("times 和 concentrations 长度必须相同")
        if len(times) < 2:
            raise ValueError("至少需要 2 个数据点")

        # 梯形法则积分
        dt = np.diff(times)
        C_mid = 0.5 * (concentrations[:-1] + concentrations[1:])

        M0 = np.sum(C_mid * dt)
        if M0 < 1e-20:
            return {'M0': 0.0, 't_mean': 0.0, 'variance': 0.0, 'skewness': 0.0}

        t_mid = 0.5 * (times[:-1] + times[1:])
        M1 = np.sum(t_mid * C_mid * dt)
        M2 = np.sum(t_mid ** 2 * C_mid * dt)
        M3 = np.sum(t_mid ** 3 * C_mid * dt)

        t_mean = M1 / M0
        variance = M2 / M0 - t_mean ** 2
        std = np.sqrt(max(variance, 0.0))

        skewness = 0.0
        if std > 1e-12:
            skewness = (M3 / M0 - 3.0 * t_mean * variance - t_mean ** 3) / (std ** 3)

        return {
            'M0': float(M0),
            't_mean': float(t_mean),
            'variance': float(variance),
            'std': float(std),
            'skewness': float(skewness)
        }

    @staticmethod
    def dispersivity_from_moments(t_mean: float, variance: float,
                                   L: float, v: float) -> float:
        """
        从突破曲线矩估算纵向弥散度

        一维对流-弥散方程的解析解（脉冲注入）给出：
            σ_t² = 2 D_L L / v³
        
        其中 D_L = α_L v + D_m，对于高流速 D_m 可忽略：
            α_L ≈ σ_t² v³ / (2 L)

        Parameters
        ----------
        t_mean : float
            平均突破时间 [s]
        variance : float
            时间方差 [s²]
        L : float
            传输距离 [m]
        v : float
            平均流速 [m/s]

        Returns
        -------
        float
            纵向弥散度 [m]
        """
        if t_mean <= 0 or L <= 0 or v <= 0:
            raise ValueError("参数必须为正")

        alpha_L = variance * v ** 3 / (2.0 * L)
        return max(alpha_L, 0.0)

    @staticmethod
    def recovery_ratio(C_out: np.ndarray, Q: np.ndarray,
                         dt: float, M_injected: float) -> float:
        """
        计算示踪剂回收率

        公式：
            R = (∫ Q(t) C_out(t) dt) / M_injected

        Parameters
        ----------
        C_out : np.ndarray
            出口浓度 [kg/m³]
        Q : np.ndarray
            出口流量 [m³/s]
        dt : float
            时间步长 [s]
        M_injected : float
            注入总质量 [kg]

        Returns
        -------
        float
            回收率
        """
        if M_injected <= 0:
            raise ValueError("M_injected 必须为正")

        recovered = np.sum(Q * C_out) * dt
        return float(recovered / M_injected)

    @staticmethod
    def peclet_number(v: float, L: float, D: float) -> float:
        """
        计算 Peclet 数

        公式：
            Pe = v L / D

        Parameters
        ----------
        v : float
            特征流速 [m/s]
        L : float
            特征长度 [m]
        D : float
            弥散系数 [m²/s]

        Returns
        -------
        float
            Peclet 数
        """
        if D <= 0:
            raise ValueError("D 必须为正")
        return v * L / D

    @staticmethod
    def reynolds_number(v: float, b: float, rho: float = 1000.0,
                        mu: float = 1.0e-3) -> float:
        """
        计算裂隙渗流 Reynolds 数

        公式：
            Re = ρ v b / μ

        Parameters
        ----------
        v : float
            流速 [m/s]
        b : float
            特征长度（裂隙开度）[m]
        rho : float
            密度 [kg/m³]
        mu : float
            粘度 [Pa·s]

        Returns
        -------
        float
            Reynolds 数
        """
        if mu <= 0 or b <= 0:
            raise ValueError("mu 和 b 必须为正")
        return rho * v * b / mu
