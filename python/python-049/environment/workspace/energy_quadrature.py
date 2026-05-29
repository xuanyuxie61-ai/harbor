#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 energy_quadrature.py
 
 融合种子项目：
   - 1147_square_integrals：正方形区域积分
   - 1318_triangle_symq_rule_original：三角形对称求积规则
   - 519_hermite_exactness：Hermite 求积精度检验
 
 科学功能：
   高精度能量积分计算与守恒检验。
   
   海啸模拟中，总能量（势能 + 动能）的守恒性是数值方法
   正确性的重要指标。本模块提供多种高精度积分方法，
   用于精确计算海啸波场的总能量，并检验数值方法的守恒性。
 
 核心物理公式：
 
   1) 海啸波场总能量：
   
      势能（单位面积）：
        E_p = (1/2) · ρ · g · η²
        
      动能（单位面积）：
        E_k = (1/2) · ρ · (h + η) · (u² + v²)
        
      总能量：
        E_total = ∫∫_Ω [E_p + E_k] dx dy
   
   2) 正方形区域单核积分（来源于 square01_monomial_integral）：
      对于单位正方形 [0,1]×[0,1] 上的单核 x^m·y^n：
      ∫∫ x^m y^n dx dy = 1/((m+1)(n+1))
      
      用于积分检验的基准值。
   
   3) 三角形对称求积规则（来源于 triangle_symq_rule）：
      在参考三角形上构造高精度对称求积规则：
      ∫∫_T f(x,y) dx dy ≈ Σ_{i=1}^N w_i · f(x_i, y_i)
      
      规则 degree 0（1 点）：
        (x, y) = (1/3, 1/3), w = 1/2
      
      通过 simplex_to_triangle 映射到任意三角形。
   
   4) Hermite 求积规则（来源于 hermite_exactness）：
      对于带权积分 ∫_{-∞}^{+∞} f(x)·exp(-x²) dx：
      
      使用 Gauss-Hermite 求积：
        ∫ f(x)·exp(-x²) dx ≈ Σ_{i=1}^n w_i · f(x_i)
      
      其中 x_i 为 Hermite 多项式的根，w_i 为对应权重。
      
      单核精确值：
        ∫_{-∞}^{+∞} x^m·exp(-x²) dx = 
          0,                         m 为奇数
          (m-1)!! · √π / 2^{m/2},   m 为偶数
      
      本模块利用此精确值检验求积规则的数值精度。
      
   5) 能量守恒检验指标：
      ε = |E(t) - E(0)| / E(0)
      
      对于保守系统，理想情况下 ε = 0。
      数值方法的 ε 应在机器精度附近。
"""

import numpy as np


class EnergyQuadrature:
    """
    能量高精度积分计算器。
    """
    
    def __init__(self, x_grid, y_grid):
        """
        Parameters
        ----------
        x_grid, y_grid : ndarray
            网格坐标（m）
        """
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.dx = x_grid[1] - x_grid[0]
        self.dy = y_grid[1] - y_grid[0]
    
    def compute_total_energy(self, eta, u, v, h_bathy, rho=1025.0, g=9.81):
        """
        计算海啸波场的总能量。
        
        E_total = ∫∫ [ (1/2)ρgη² + (1/2)ρ(h+η)(u²+v²) ] dx dy
        
        Parameters
        ----------
        eta, u, v : ndarray
            海面高度和流速场
        h_bathy : ndarray
            静水深场
        rho : float
            海水密度
        g : float
            重力加速度
            
        Returns
        -------
        E_total : float
            总能量（J）
        """
        # ============================================
        # [HOLE 2] 海啸波场总能量计算
        # ============================================
        # 此处被挖空，需要实现以下核心科学计算：
        #
        # 1. 势能密度：E_p = (1/2)·ρ·g·η²
        #    （η 为海面高度相对于静水面的位移）
        #
        # 2. 动能密度：E_k = (1/2)·ρ·(h+η)·(u²+v²)
        #    （h 为静水深，u/v 为水平流速分量）
        #    注意：H = h_bathy + eta 需保证 H > 0
        #
        # 3. 总面积分：
        #    E_total = ∫∫ (E_p + E_k) dx dy
        #    在均匀网格上可用矩形法则/梯形法则近似
        #
        # 物理约束：该能量公式必须与 tsunami_pde_solver.py
        # 中的分裂格式一致。_solver 中压强梯度为 -g·∂η/∂x，
        # 驱动速度场演化，对应势能-动能转换系数为 g。
        # 若 solver 中改变了 g 的使用方式或速度定义，
        # 此处的能量公式必须同步调整。
        # ============================================
        E_total = 0.0
        
        return E_total
    
    def compute_energy_flux(self, eta, u, v, h_bathy, rho=1025.0, g=9.81):
        """
        计算能量通量（能量传输率）。
        
        F = ∫∫ ρ g η (h+η) u · n dx dy
        """
        H = h_bathy + eta
        flux_x = rho * g * eta * H * u
        flux_y = rho * g * eta * H * v
        
        # 边界通量
        flux_left = np.sum(flux_x[:, 0]) * self.dy
        flux_right = np.sum(flux_x[:, -1]) * self.dy
        flux_bottom = np.sum(flux_y[0, :]) * self.dx
        flux_top = np.sum(flux_y[-1, :]) * self.dx
        
        return flux_left, flux_right, flux_bottom, flux_top
    
    # ============================================================
    # 正方形区域积分（来源于 square_integrals）
    # ============================================================
    
    def square_monomial_integral(self, exponents):
        """
        计算单位正方形 [0,1]×[0,1] 上的单核积分。
        
        ∫∫ x^{e1} y^{e2} dx dy = 1 / ((e1+1)(e2+1))
        
        Parameters
        ----------
        exponents : tuple
            (e1, e2)
            
        Returns
        -------
        integral : float
            精确积分值
        """
        e1, e2 = exponents
        
        if e1 < 0 or e2 < 0:
            raise ValueError("指数必须非负")
        
        integral = 1.0 / ((e1 + 1) * (e2 + 1))
        return integral
    
    def test_square_quadrature(self, max_degree=5):
        """
        使用单核积分检验正方形区域求积精度。
        """
        errors = {}
        for m in range(max_degree + 1):
            for n in range(max_degree + 1):
                if m + n <= max_degree:
                    exact = self.square_monomial_integral((m, n))
                    # 数值积分（矩形法则）
                    numerical = self._numerical_square_integral(m, n)
                    error = abs(numerical - exact)
                    errors[(m, n)] = error
        return errors
    
    def _numerical_square_integral(self, m, n):
        """
        使用数值方法计算单核积分。
        """
        N = 100
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        dx = 1.0 / (N - 1)
        dy = 1.0 / (N - 1)
        
        integral = 0.0
        for yi in y:
            for xi in x:
                integral += (xi**m) * (yi**n) * dx * dy
        
        return integral
    
    # ============================================================
    # 三角形对称求积规则（来源于 triangle_symq_rule）
    # ============================================================
    
    def triangle_symmetric_quadrature(self, f, triangle_vertices, rule_degree=0):
        """
        使用对称求积规则在三角形上积分。
        
        Parameters
        ----------
        f : callable
            被积函数 f(x, y)
        triangle_vertices : ndarray, shape (3, 2)
            三角形三个顶点
        rule_degree : int
            求积规则阶数（当前实现 degree 0）
            
        Returns
        -------
        integral : float
            积分近似值
        """
        v1, v2, v3 = triangle_vertices
        
        # 参考三角形上的求积节点和权重
        if rule_degree == 0:
            # 1 点规则（重心）
            xi_nodes = np.array([1.0/3.0])
            eta_nodes = np.array([1.0/3.0])
            weights = np.array([0.5])  # 参考三角形面积 = 1/2
        else:
            # 默认使用 degree 0
            xi_nodes = np.array([1.0/3.0])
            eta_nodes = np.array([1.0/3.0])
            weights = np.array([0.5])
        
        # 映射到物理三角形
        integral = 0.0
        for i in range(len(weights)):
            # simplex_to_triangle 映射
            s = np.array([xi_nodes[i], eta_nodes[i]])
            t = self._simplex_to_triangle(v1, v2, v3, s)
            
            # 雅可比行列式（三角形面积因子）
            J = abs((v2[0]-v1[0])*(v3[1]-v1[1]) - (v3[0]-v1[0])*(v2[1]-v1[1]))
            
            integral += weights[i] * f(t[0], t[1]) * J
        
        return integral
    
    def _simplex_to_triangle(self, v1, v2, v3, s):
        """
        将参考单纯形 [0,0], [1,0], [0,1] 映射到物理三角形。
        
        t = v1 * (1 - s1 - s2) + v2 * s1 + v3 * s2
        """
        t = v1 * (1.0 - s[0] - s[1]) + v2 * s[0] + v3 * s[1]
        return t
    
    def triangle_symmetric_quadrature_test(self):
        """
        测试三角形对称求积规则。
        
        在标准三角形 (0,0), (1,0), (0,1) 上积分 f(x,y) = 1，
        精确值为 0.5。
        """
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        f = lambda x, y: 1.0
        return self.triangle_symmetric_quadrature(f, triangle, rule_degree=0)
    
    # ============================================================
    # Hermite 求积精度检验（来源于 hermite_exactness）
    # ============================================================
    
    def hermite_integral_exact(self, n, option=1):
        """
        计算 Hermite 单核积分的精确值。
        
        H(n, option) = ∫_{-∞}^{+∞} x^n · exp(-x²) dx
        
        精确解：
          n 为奇数：0
          n 为偶数：(n-1)!! · √π / 2^{n/2}
        
        Parameters
        ----------
        n : int
            单核阶数
        option : int
            权函数选项（1: exp(-x²)）
            
        Returns
        -------
        value : float
            精确积分值
        """
        if n < 0:
            return 0.0
        
        if n % 2 == 1:
            return 0.0
        
        if option == 1:
            # (n-1)!! · √π / 2^{n/2}
            return self._double_factorial(n - 1) * np.sqrt(np.pi) / (2.0 ** (n / 2.0))
        else:
            return self._double_factorial(n - 1) * np.sqrt(np.pi) / (2.0 ** (n / 2.0))
    
    def _double_factorial(self, n):
        """
        计算双阶乘 n!!。
        """
        if n < 1:
            return 1.0
        
        value = 1.0
        while n > 1:
            value *= n
            n -= 2
        return value
    
    def gauss_hermite_quadrature(self, f, n_points):
        """
        Gauss-Hermite 求积。
        
        ∫ f(x)·exp(-x²) dx ≈ Σ w_i · f(x_i)
        """
        # 使用 numpy 的 Hermite 多项式根和权重
        from numpy.polynomial.hermite import hermgauss
        
        x, w = hermgauss(n_points)
        result = np.sum(w * f(x))
        return result
    
    def test_hermite_exactness(self, max_degree=9, n_points=10):
        """
        检验 Gauss-Hermite 求积规则的精确度。
        
        来源于 hermite_exactness 的核心思想。
        
        Gauss-Hermite 求积使用 n 个点，应能精确积分
        阶数不超过 2n-1 的单核。
        
        Parameters
        ----------
        max_degree : int
            检验的最大阶数
        n_points : int
            求积点数
            
        Returns
        -------
        errors : dict
            各阶单核的求积误差
        """
        errors = {}
        
        for degree in range(max_degree + 1):
            exact = self.hermite_integral_exact(degree, option=1)
            
            # 数值求积
            f = lambda x: x ** degree
            numerical = self.gauss_hermite_quadrature(f, n_points)
            
            if exact == 0.0:
                error = abs(numerical)
            else:
                error = abs(numerical - exact) / abs(exact)
            
            errors[degree] = error
        
        return errors
    
    def compute_energy_by_triangle_quadrature(self, eta, u, v, h_bathy,
                                               rho=1025.0, g=9.81):
        """
        使用三角形求积计算能量（将网格划分为三角形单元）。
        
        每个矩形网格单元划分为两个三角形：
        T1: (i,j), (i+1,j), (i,j+1)
        T2: (i+1,j), (i+1,j+1), (i,j+1)
        """
        E_total = 0.0
        
        for j in range(len(self.y_grid) - 1):
            for i in range(len(self.x_grid) - 1):
                # 四个顶点坐标
                p1 = np.array([self.x_grid[i], self.y_grid[j]])
                p2 = np.array([self.x_grid[i+1], self.y_grid[j]])
                p3 = np.array([self.x_grid[i], self.y_grid[j+1]])
                p4 = np.array([self.x_grid[i+1], self.y_grid[j+1]])
                
                # 顶点处的能量密度
                def energy_density(x, y):
                    # 双线性插值
                    xi = (x - self.x_grid[i]) / self.dx
                    eta_val = (y - self.y_grid[j]) / self.dy
                    
                    e00 = 0.5 * rho * g * eta[j, i]**2
                    e10 = 0.5 * rho * g * eta[j, i+1]**2
                    e01 = 0.5 * rho * g * eta[j+1, i]**2
                    e11 = 0.5 * rho * g * eta[j+1, i+1]**2
                    
                    val = (1-xi)*(1-eta_val)*e00 + xi*(1-eta_val)*e10 + \
                          (1-xi)*eta_val*e01 + xi*eta_val*e11
                    return val
                
                # 三角形 T1
                E_total += self.triangle_symmetric_quadrature(
                    energy_density, np.array([p1, p2, p3]), rule_degree=0
                )
                
                # 三角形 T2
                E_total += self.triangle_symmetric_quadrature(
                    energy_density, np.array([p2, p4, p3]), rule_degree=0
                )
        
        return E_total
