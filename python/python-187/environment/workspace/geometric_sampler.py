#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma


class GeometricSampler:
    
    def __init__(self, n_samples=5000):
        self.n_samples = max(n_samples, 10)
        self.rng = np.random.RandomState(42)
    
    def sample_unit_circle(self, n):
        n = max(n, 1)
        theta = self.rng.rand(n) * 2.0 * np.pi
        x = np.column_stack([np.cos(theta), np.sin(theta)])
        return x
    
    def sample_positive_circle(self, n):
        n = max(n, 1)
        theta = self.rng.rand(n) * 2.0 * np.pi
        x = np.abs(np.cos(theta))
        y = np.abs(np.sin(theta))
        return np.column_stack([x, y])
    
    def circle_monomial_integral(self, e):
        e = np.asarray(e, dtype=int)
        if np.any(e < 0):
            return 0.0
        
        if np.any(e % 2 == 1):
            return 0.0
        
        integral = 2.0
        for i in range(2):
            integral *= gamma(0.5 * (e[i] + 1))
        integral /= gamma(0.5 * (e[0] + e[1] + 2))
        
        return float(integral)
    
    def monte_carlo_circle_integral(self, func, n):
        n = max(n, 10)
        samples = self.sample_unit_circle(n)
        vals = np.array([func(p[0], p[1]) for p in samples])
        return (2.0 * np.pi / n) * np.sum(vals)
    
    def positive_circle_distance_stats(self, n):
        n = max(n, 10)
        p = self.sample_positive_circle(n)
        q = self.sample_positive_circle(n)
        dists = np.linalg.norm(p - q, axis=1)
        
        mu = np.mean(dists)
        if n > 1:
            var = np.sum((dists - mu) ** 2) / (n - 1)
        else:
            var = 0.0
        
        return float(mu), float(var)
    
    def parallelogram_area_3d(self, p):
        p = np.asarray(p, dtype=float)
        if p.shape[0] < 3:
            return 0.0
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        cross = np.cross(v1, v2)
        return float(np.linalg.norm(cross))
    
    def quadrilateral_area_3d(self, q):
        q = np.asarray(q, dtype=float)
        m = q.shape[0]
        if m < 3:
            return 0.0
        if m == 3:

            v1 = q[1] - q[0]
            v2 = q[2] - q[0]
            return 0.5 * np.linalg.norm(np.cross(v1, v2))
        

        p = np.zeros((4, 3))
        for i in range(3):
            p[i] = (q[i] + q[i+1]) / 2.0
        p[3] = (q[3] + q[0]) / 2.0
        
        para_area = self.parallelogram_area_3d(p)
        return 2.0 * para_area
    
    def plane_tetrahedron_intersect(self, pp, normal, t):
        pp = np.asarray(pp, dtype=float).flatten()
        normal = np.asarray(normal, dtype=float).flatten()
        t = np.asarray(t, dtype=float)
        
        if t.shape[0] != 4 or t.shape[1] != 3:
            return 0, np.zeros((4, 3))
        
        dn = np.linalg.norm(normal)
        if dn < 1e-12:
            return 0, np.zeros((4, 3))
        
        n_unit = normal / dn
        

        d = np.array([np.dot(n_unit, t[i] - pp) for i in range(4)])
        

        if np.all(d < -1e-12) or np.all(d > 1e-12):
            return 0, np.zeros((4, 3))
        
        pint = np.zeros((4, 3))
        int_num = 0
        
        for j1 in range(4):
            if abs(d[j1]) < 1e-12:

                pint[int_num] = t[j1]
                int_num += 1
            else:
                for j2 in range(j1 + 1, 4):

                    if d[j1] * d[j2] < -1e-12:
                        pint[int_num] = (
                            d[j1] * t[j2] - d[j2] * t[j1]
                        ) / (d[j1] - d[j2])
                        int_num += 1
        

        int_num = min(int_num, 4)
        

        if int_num == 4:
            area1 = self.quadrilateral_area_3d(pint)
            pint2 = np.copy(pint)
            pint2[2], pint2[3] = pint[3].copy(), pint[2].copy()
            area2 = self.quadrilateral_area_3d(pint2)
            if area2 > area1:
                pint = pint2
        
        return int_num, pint
