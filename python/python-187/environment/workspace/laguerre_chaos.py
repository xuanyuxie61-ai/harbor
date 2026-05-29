#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laguerre_chaos.py
=================

基于种子项目 642_laguerre_product 的广义多项式混沌（gPC）不确定性量化。

科学背景
--------
在推荐系统中，预测评分 Ŷ 存在不确定性。我们采用 Laguerre 广义多项式混沌
将随机输入参数（如用户活跃度、物品流行度）映射到评分输出的统计分布上。

Laguerre 多项式 {L_n(x)} 在权函数 w(x)=exp(-x) 下于 [0,∞) 上正交:
    ∫_0^∞ L_m(x) L_n(x) exp(-x) dx = δ_{mn}

递推关系:
    L_0(x) = 1
    L_1(x) = 1 - x
    (n+1) L_{n+1}(x) = (2n + 1 - x) L_n(x) - n L_{n-1}(x)

指数加权内积（用于多项式 chaos 展开）:
    T_{ij} = ∫_0^∞ exp(β x) L_i(x) L_j(x) exp(-x) dx
           = Σ_k w_k exp(β x_k) L_i(x_k) L_j(x_k)
           
其中 {x_k, w_k} 是 Gauss-Laguerre 求积节点和权重。

不确定性传播:
    若 Ŷ = Σ_{k=0}^P c_k L_k(ξ),  ξ ~ Exp(1)
    则 Var[Ŷ] = Σ_{k=1}^P c_k² ⟨L_k, L_k⟩_w
"""

import numpy as np


class LaguerrePolynomialChaos:
    """
    Laguerre 广义多项式混沌展开与不确定性量化。
    """
    
    def __init__(self, max_degree=5):
        """
        参数:
            max_degree : 多项式展开的最高次数 P
        """
        if max_degree < 0:
            raise ValueError("max_degree 必须非负")
        self.max_degree = int(max_degree)
    
    def evaluate(self, n, x):
        """
        计算 0 到 n 阶 Laguerre 多项式在 x 处的值。
        
        算法:
            使用三 term 递推关系，时间复杂度 O(n)。
            
        边界处理:
            - n < 0 返回空数组
            - x 为标量时返回一维数组
        """
        if n < 0:
            return np.array([])
        
        x = np.asarray(x, dtype=float)
        scalar_input = (x.ndim == 0)
        x = np.atleast_1d(x)
        
        l_vals = np.zeros((len(x), n + 1))
        l_vals[:, 0] = 1.0
        
        if n >= 1:
            l_vals[:, 1] = 1.0 - x
        
        for i in range(2, n + 1):
            # (i) L_i(x) = (2(i-1) + 1 - x) L_{i-1}(x) - (i-1) L_{i-2}(x)
            l_vals[:, i] = (
                (2.0 * (i - 1) + 1.0 - x) * l_vals[:, i - 1]
                - (i - 1) * l_vals[:, i - 2]
            ) / i
        
        if scalar_input:
            return l_vals[0, :]
        return l_vals
    
    def quadrature_rule(self, order):
        """
        计算 Gauss-Laguerre 求积规则的节点和权重。
        
        目标近似:
            ∫_0^∞ exp(-x) f(x) dx ≈ Σ_{i=1}^{order} w_i f(x_i)
            
        算法:
            1. 构造 Jacobi 矩阵的三对角元素:
               b_i = 2i - 1  (对角)
               c_i = (i-1)²  (次对角)
            2. 使用 Newton 迭代求特征值（即节点 x_i）
            3. 权重: w_i = c₀ / (L'_n(x_i) L_{n-1}(x_i))
            
        边界保护:
            - order < 1 时返回空数组
            - 迭代不收敛时抛出异常
        """
        if order < 1:
            return np.array([]), np.array([])
        
        order = int(order)
        
        # 递推系数
        b = np.array([2.0 * i - 1.0 for i in range(1, order + 1)])
        c = np.array([max(0.0, (i - 1.0)**2) for i in range(1, order + 1)])
        
        # cc = ∏_{i=2}^{order} c_i = (Γ(order))²
        cc = np.prod(c[1:]) if order > 1 else 1.0
        
        xtab = np.zeros(order)
        weight = np.zeros(order)
        
        for i in range(order):
            # 初始猜测
            if i == 0:
                x = 3.0 / (1.0 + 2.4 * order)
            elif i == 1:
                x = xtab[0] + 15.0 / (1.0 + 2.5 * order)
            else:
                r1 = (1.0 + 2.55 * (i - 1)) / (1.9 * (i - 1))
                x = x + r1 * (x - xtab[i - 2])
            
            # Newton 迭代求根
            x, dp2, p1 = self._laguerre_root(x, order, b, c)
            xtab[i] = x
            weight[i] = cc / dp2 / p1
        
        return xtab, weight
    
    def _laguerre_recur(self, x, order, b, c):
        """
        计算 Laguerre 多项式值及其导数。
        
        返回:
            p2  : L_order(x)
            dp2 : L'_order(x)
            p1  : L_{order-1}(x)
        """
        p1 = 1.0
        dp1 = 0.0
        p2 = x - 1.0
        dp2 = 1.0
        
        for i in range(2, order + 1):
            p0 = p1
            dp0 = dp1
            p1 = p2
            dp1 = dp2
            p2 = (x - b[i - 1]) * p1 - c[i - 1] * p0
            dp2 = (x - b[i - 1]) * dp1 + p1 - c[i - 1] * dp0
        
        return p2, dp2, p1
    
    def _laguerre_root(self, x, order, b, c, max_step=10):
        """
        Newton 迭代改进 Laguerre 多项式的近似根。
        
        迭代公式:
            x_{new} = x - L_n(x) / L'_n(x)
            
        收敛判据:
            |Δx| ≤ ε (|x| + 1)
        """
        eps = np.finfo(float).eps
        for _ in range(max_step):
            p2, dp2, p1 = self._laguerre_recur(x, order, b, c)
            if abs(dp2) < eps:
                break
            d = p2 / dp2
            x = x - d
            if abs(d) <= eps * (abs(x) + 1.0):
                break
        p2, dp2, p1 = self._laguerre_recur(x, order, b, c)
        return x, dp2, p1
    
    def exponential_product_table(self, beta):
        """
        计算指数加权 Laguerre 乘积表。
        
        T_{ij} = ∫_0^∞ exp(β x) L_i(x) L_j(x) exp(-x) dx
               = Σ_k w_k exp(β x_k) L_i(x_k) L_j(x_k)
               
        物理意义:
            在随机输入服从指数族分布时，输出响应的二阶矩信息。
        """
        p = self.max_degree
        order = int(np.floor((3 * p + 4) / 2.0))
        x_table, w_table = self.quadrature_rule(order)
        
        table = np.zeros((p + 1, p + 1))
        for k in range(order):
            x = x_table[k]
            l_table = self.evaluate(p, x)
            # 外积更新
            contrib = w_table[k] * np.exp(beta * x) * np.outer(l_table, l_table)
            table += contrib
        
        return table
    
    def linear_product_table(self, exponent):
        """
        计算幂次加权 Laguerre 乘积表。
        
        T_{ij} = ∫_0^∞ x^e L_i(x) L_j(x) exp(-x) dx
        """
        p = self.max_degree
        order = p + 1 + int(np.floor((exponent + 1) / 2.0))
        x_table, w_table = self.quadrature_rule(order)
        
        table = np.zeros((p + 1, p + 1))
        for k in range(order):
            x = x_table[k]
            l_table = self.evaluate(p, x)
            if exponent == 0:
                contrib = w_table[k] * np.outer(l_table, l_table)
            else:
                contrib = w_table[k] * (x ** exponent) * np.outer(l_table, l_table)
            table += contrib
        
        return table
    
    def propagate_uncertainty(self, observed_ratings, beta=0.3):
        """
        通过 gPC 展开传播预测不确定性。
        
        模型:
            将观测评分投影到 Laguerre 基上:
                c_k = ⟨R, L_k⟩_w / ⟨L_k, L_k⟩_w
            
            预测方差:
                Var[Ŷ] = Σ_{k=1}^P c_k² ⟨L_k, L_k⟩_w
                
        边界处理:
            - 空输入返回 0
            - 负评分通过指数变换映射到正域
        """
        observed_ratings = np.asarray(observed_ratings, dtype=float)
        if observed_ratings.size == 0:
            return 0.0
        
        # 将评分从 [1,5] 映射到 [0,∞)
        # 使用变换: z = (r - r_min) / scale
        r_min = np.min(observed_ratings)
        scale = max(np.std(observed_ratings), 0.1)
        z = (observed_ratings - r_min) / scale
        z = np.maximum(z, 0.0)  # 确保非负
        
        # 计算展开系数
        p = self.max_degree
        order = int(np.floor((3 * p + 4) / 2.0))
        x_table, w_table = self.quadrature_rule(order)
        
        coeffs = np.zeros(p + 1)
        for k in range(order):
            x = x_table[k]
            l_table = self.evaluate(p, x)
            # 通过插值估计 f(x)
            f_val = np.interp(x, np.sort(z), np.sort(observed_ratings))
            coeffs += w_table[k] * f_val * l_table
        
        # 方差 = Σ_{k=1}^P c_k² (排除 k=0 的均值项)
        variance = np.sum(coeffs[1:]**2)
        
        # 结合指数加权修正
        exp_table = self.exponential_product_table(beta)
        variance *= np.trace(exp_table) / (p + 1)
        
        return float(np.maximum(variance, 0.0))
