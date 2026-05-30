#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class HeatDiffusionRecommender:
    
    def __init__(self, alpha=0.05, n_steps=8):
        self.alpha = max(alpha, 1e-6)
        self.n_steps = max(n_steps, 1)
    
    def _build_laplacian(self, R_obs):
        n, m = R_obs.shape
        h = 1.0
        

        def apply_laplacian(u):
            Lu = np.zeros_like(u)

            Lu[1:-1, 1:-1] = (
                u[2:, 1:-1] + u[:-2, 1:-1] +
                u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
            ) / (h ** 2)

            Lu[0, 1:-1] = (u[1, 1:-1] + u[0, 2:] + u[0, :-2] - 3.0 * u[0, 1:-1]) / (h ** 2)
            Lu[-1, 1:-1] = (u[-2, 1:-1] + u[-1, 2:] + u[-1, :-2] - 3.0 * u[-1, 1:-1]) / (h ** 2)
            Lu[1:-1, 0] = (u[2:, 0] + u[:-2, 0] + u[1:-1, 1] - 3.0 * u[1:-1, 0]) / (h ** 2)
            Lu[1:-1, -1] = (u[2:, -1] + u[:-2, -1] + u[1:-1, -2] - 3.0 * u[1:-1, -1]) / (h ** 2)

            Lu[0, 0] = (u[1, 0] + u[0, 1] - 2.0 * u[0, 0]) / (h ** 2)
            Lu[0, -1] = (u[1, -1] + u[0, -2] - 2.0 * u[0, -1]) / (h ** 2)
            Lu[-1, 0] = (u[-2, 0] + u[-1, 1] - 2.0 * u[-1, 0]) / (h ** 2)
            Lu[-1, -1] = (u[-2, -1] + u[-1, -2] - 2.0 * u[-1, -1]) / (h ** 2)
            return Lu
        
        return apply_laplacian
    
    def _solve_linear_system(self, A_op, b, mask_known, u_init, max_iter=100, tol=1e-6):
        u = np.copy(u_init)

        u[mask_known] = b[mask_known]
        
        for _ in range(max_iter):
            u_old = np.copy(u)


            residual = b - u_old + self.alpha * self._laplacian_op(u_old) * self.dt
            diag = 1.0 + 4.0 * self.alpha * self.dt
            u = u_old + residual / diag
            

            u[mask_known] = b[mask_known]
            
            if np.linalg.norm(u - u_old) < tol:
                break
        
        return u
    
    def diffuse(self, R_obs):
        R = np.array(R_obs, dtype=float)
        n, m = R.shape
        
        if n == 0 or m == 0:
            return R
        
        mask_known = ~np.isnan(R)
        if not np.any(mask_known):
            return np.full_like(R, 3.0)
        
        global_mean = np.nanmean(R)
        if np.isnan(global_mean):
            global_mean = 3.0
        

        u = np.full((n, m), global_mean, dtype=float)
        u[mask_known] = R[mask_known]
        

        self.dt = 0.5 / max(self.n_steps, 1)
        

        self._laplacian_op = self._build_laplacian(R)
        

        for step in range(self.n_steps):

            source = np.zeros_like(u)
            source[mask_known] = (R[mask_known] - u[mask_known]) * 0.5
            rhs = u + self.dt * source
            

            u_new = np.copy(u)
            for _ in range(50):
                u_old = np.copy(u_new)
                Lu = self._laplacian_op(u_old)






                u_new = u_old
                u_new[mask_known] = R[mask_known]
                
                if np.linalg.norm(u_new - u_old) < 1e-5:
                    break
            
            u = u_new
        

        u = np.clip(u, 1.0, 5.0)
        return u
