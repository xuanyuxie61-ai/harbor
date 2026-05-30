#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class CordicEngine:
    
    def __init__(self, n_iter=24):
        self.n_iter = max(n_iter, 1)
        

        self.angles = np.array([
            np.arctan(2.0 ** (-i)) for i in range(60)
        ], dtype=float)
        


        self.kprod = np.zeros(40)
        k = 1.0
        for i in range(40):
            k *= 1.0 / np.sqrt(1.0 + 2.0 ** (-2.0 * i))
            self.kprod[i] = k
        

        self.exp_table = np.array([
            np.exp(2.0 ** (-(i + 1))) for i in range(30)
        ], dtype=float)
    
    def cossin(self, beta):
        beta = float(beta)
        

        theta = self._angle_shift(beta, -np.pi)
        

        sign_factor = 1.0
        if theta < -0.5 * np.pi:
            theta += np.pi
            sign_factor = -1.0
        elif theta > 0.5 * np.pi:
            theta -= np.pi
            sign_factor = -1.0
        
        x, y = 1.0, 0.0
        angle = self.angles[0]
        
        for j in range(self.n_iter):
            sigma = -1.0 if theta < 0.0 else 1.0
            factor = sigma * (2.0 ** (-j))
            
            x_new = x - factor * y
            y_new = factor * x + y
            x, y = x_new, y_new
            
            theta -= sigma * angle
            
            if j + 1 < len(self.angles):
                angle = self.angles[j + 1]
            else:
                angle /= 2.0
        

        if self.n_iter > 0:
            k_factor = self.kprod[min(self.n_iter - 1, len(self.kprod) - 1)]
            x *= k_factor
            y *= k_factor
        
        x *= sign_factor
        y *= sign_factor
        
        return x, y
    
    def exp_cordic(self, x):
        x = float(x)
        

        if x > 700:
            return float('inf')
        if x < -700:
            return 0.0
        
        e_base = np.e
        x_int = int(np.floor(x))
        z = x - x_int
        

        poweroftwo = 0.5
        fx = 1.0
        for i in range(self.n_iter):
            if poweroftwo < z:
                if i < len(self.exp_table):
                    ai = self.exp_table[i]
                else:
                    ai = 1.0 + (self.exp_table[-1] - 1.0) * (2.0 ** (-(i - len(self.exp_table) + 1)))
                fx *= ai
                z -= poweroftwo
            poweroftwo /= 2.0
        

        fx *= (1.0 + z * (1.0 + z / 2.0 * (1.0 + z / 3.0 * (1.0 + z / 4.0))))
        

        if x_int < 0:
            for _ in range(-x_int):
                fx /= e_base
        else:
            for _ in range(x_int):
                fx *= e_base
        
        return fx
    
    def log_cordic(self, x):
        x = float(x)
        if x <= 0.0:
            return float('-inf')
        
        e_base = np.e
        k = 0
        while x >= e_base:
            k += 1
            x /= e_base
        while x < 1.0:
            k -= 1
            x *= e_base
        

        poweroftwo = 0.5
        total = 0.0
        for i in range(self.n_iter):
            if i < len(self.exp_table):
                ai = self.exp_table[i]
            else:
                ai = 1.0 + (self.exp_table[-1] - 1.0) / 2.0
            if ai < x:
                total += poweroftwo
                x /= ai
            poweroftwo /= 2.0
        

        x -= 1.0
        x = x * (1.0 - x / 2.0 * (1.0 + x / 3.0 * (1.0 - x / 4.0)))
        
        return k + total + x
    
    def sqrt_cordic(self, x):
        x = float(x)
        if x < 0.0:
            return float('nan')
        if x == 0.0:
            return 0.0
        if x == 1.0:
            return 1.0
        
        poweroftwo = 1.0
        if x < 1.0:
            while poweroftwo * poweroftwo > x:
                poweroftwo /= 2.0
            y = poweroftwo
        else:
            while poweroftwo * poweroftwo <= x:
                poweroftwo *= 2.0
            y = poweroftwo / 2.0
        
        for _ in range(self.n_iter):
            poweroftwo /= 2.0
            if (y + poweroftwo) ** 2 <= x:
                y += poweroftwo
        
        return y
    
    def compute_similarity_kernel(self, P, Q, sigma):
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)
        
        n_users = P.shape[0]
        n_items = Q.shape[0]
        


        p_norm = np.sum(P**2, axis=1)
        q_norm = np.sum(Q**2, axis=1)
        dot = P @ Q.T
        dist_sq = p_norm[:, None] + q_norm[None, :] - 2.0 * dot
        dist_sq = np.maximum(dist_sq, 0.0)
        


        K = np.exp(-dist_sq / (2.0 * sigma**2))
        return K
    
    def _angle_shift(self, alpha, beta):
        two_pi = 2.0 * np.pi
        gamma = alpha - beta
        gamma = gamma - two_pi * np.floor(gamma / two_pi)
        return beta + gamma
