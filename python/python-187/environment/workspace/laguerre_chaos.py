#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class LaguerrePolynomialChaos:
    
    def __init__(self, max_degree=5):
        if max_degree < 0:
            raise ValueError("max_degree 必须非负")
        self.max_degree = int(max_degree)
    
    def evaluate(self, n, x):
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

            l_vals[:, i] = (
                (2.0 * (i - 1) + 1.0 - x) * l_vals[:, i - 1]
                - (i - 1) * l_vals[:, i - 2]
            ) / i
        
        if scalar_input:
            return l_vals[0, :]
        return l_vals
    
    def quadrature_rule(self, order):
        if order < 1:
            return np.array([]), np.array([])
        
        order = int(order)
        

        b = np.array([2.0 * i - 1.0 for i in range(1, order + 1)])
        c = np.array([max(0.0, (i - 1.0)**2) for i in range(1, order + 1)])
        

        cc = np.prod(c[1:]) if order > 1 else 1.0
        
        xtab = np.zeros(order)
        weight = np.zeros(order)
        
        for i in range(order):

            if i == 0:
                x = 3.0 / (1.0 + 2.4 * order)
            elif i == 1:
                x = xtab[0] + 15.0 / (1.0 + 2.5 * order)
            else:
                r1 = (1.0 + 2.55 * (i - 1)) / (1.9 * (i - 1))
                x = x + r1 * (x - xtab[i - 2])
            

            x, dp2, p1 = self._laguerre_root(x, order, b, c)
            xtab[i] = x
            weight[i] = cc / dp2 / p1
        
        return xtab, weight
    
    def _laguerre_recur(self, x, order, b, c):
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
        p = self.max_degree
        order = int(np.floor((3 * p + 4) / 2.0))
        x_table, w_table = self.quadrature_rule(order)
        
        table = np.zeros((p + 1, p + 1))
        for k in range(order):
            x = x_table[k]
            l_table = self.evaluate(p, x)

            contrib = w_table[k] * np.exp(beta * x) * np.outer(l_table, l_table)
            table += contrib
        
        return table
    
    def linear_product_table(self, exponent):
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
        observed_ratings = np.asarray(observed_ratings, dtype=float)
        if observed_ratings.size == 0:
            return 0.0
        


        r_min = np.min(observed_ratings)
        scale = max(np.std(observed_ratings), 0.1)
        z = (observed_ratings - r_min) / scale
        z = np.maximum(z, 0.0)
        

        p = self.max_degree
        order = int(np.floor((3 * p + 4) / 2.0))
        x_table, w_table = self.quadrature_rule(order)
        
        coeffs = np.zeros(p + 1)
        for k in range(order):
            x = x_table[k]
            l_table = self.evaluate(p, x)

            f_val = np.interp(x, np.sort(z), np.sort(observed_ratings))
            coeffs += w_table[k] * f_val * l_table
        

        variance = np.sum(coeffs[1:]**2)
        

        exp_table = self.exponential_product_table(beta)
        variance *= np.trace(exp_table) / (p + 1)
        
        return float(np.maximum(variance, 0.0))
