#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class BathymetryGenerator:
    
    def __init__(self, x_grid, y_grid):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.nx = len(x_grid)
        self.ny = len(y_grid)
        self.Lx = x_grid[-1] - x_grid[0]
        self.Ly = y_grid[-1] - y_grid[0]
    
    def generate_random_bathymetry(self, depth_mean=4000.0, depth_std=800.0,
                                    continental_slope=True,
                                    hurst_exponent=0.8):

        h_random = self._von_karman_terrain(
            depth_mean, depth_std, hurst_exponent
        )
        

        if continental_slope:
            h_random = self._add_continental_slope(h_random, depth_mean)
        

        h_random = np.maximum(h_random, 10.0)
        

        h_random = self._adjust_marginal_distributions(h_random, depth_mean, depth_std)
        
        return h_random
    
    def _von_karman_terrain(self, depth_mean, depth_std, H):

        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.Lx / self.nx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.Ly / self.ny)
        KX, KY = np.meshgrid(kx, ky)
        k_mag = np.sqrt(KX**2 + KY**2)
        k_mag[0, 0] = 1e-6
        

        k_0 = 2.0 * np.pi / max(self.Lx, self.Ly)
        

        P = (k_mag**2 + k_0**2) ** (-H - 1.0)
        P[0, 0] = 0.0
        

        P = P / np.sum(P)
        

        phase_seed = self._unicycle_random_phase(self.nx * self.ny)
        np.random.seed(int(np.sum(phase_seed[:10])) % 2**31)
        

        amplitude = np.sqrt(P / 2.0)
        N1 = np.random.randn(self.ny, self.nx)
        N2 = np.random.randn(self.ny, self.nx)
        

        h_hat = amplitude * (N1 + 1j * N2)
        h_hat = self._enforce_conjugate_symmetry(h_hat)
        

        h = np.real(np.fft.ifft2(h_hat))
        

        h = (h - np.mean(h)) / (np.std(h) + 1e-12) * depth_std + depth_mean
        
        return h
    
    def _unicycle_random_phase(self, n):

        u = np.arange(1, n + 1)
        for i in range(1, n):
            j = np.random.randint(i, n)
            u[i], u[j] = u[j], u[i]
        return u
    
    def _enforce_conjugate_symmetry(self, h_hat):
        ny, nx = h_hat.shape
        h_hat_sym = h_hat.copy()
        
        for j in range(ny):
            for i in range(nx):
                j_conj = (-j) % ny
                i_conj = (-i) % nx
                if j == 0 and i == 0:
                    h_hat_sym[j, i] = np.real(h_hat_sym[j, i])
                else:
                    val = 0.5 * (h_hat[j, i] + np.conj(h_hat[j_conj, i_conj]))
                    h_hat_sym[j, i] = val
                    h_hat_sym[j_conj, i_conj] = np.conj(val)
        
        return h_hat_sym
    
    def _add_continental_slope(self, h, depth_mean):
        y_norm = (self.y_grid - self.y_grid[0]) / (self.y_grid[-1] - self.y_grid[0])
        

        h_shelf = 200.0
        h_deep = depth_mean * 1.2
        y_slope = 0.6
        w_slope = 0.08
        

        profile = h_deep - (h_deep - h_shelf) * (1.0 - np.tanh((y_norm - y_slope) / w_slope)) / 2.0
        

        h_new = h.copy()
        for j in range(self.ny):
            h_new[j, :] = h[j, :] + (profile[j] - depth_mean)
        
        return h_new
    
    def _adjust_marginal_distributions(self, h, depth_mean, depth_std):
        h_adj = h.copy()
        

        target_row = np.ones(self.ny) * depth_mean * self.nx

        target_col = np.ones(self.nx) * depth_mean * self.ny
        

        for _ in range(10):

            row_sum = np.sum(h_adj, axis=1)
            row_factor = target_row / (row_sum + 1e-12)
            h_adj = h_adj * row_factor[:, np.newaxis]
            

            col_sum = np.sum(h_adj, axis=0)
            col_factor = target_col / (col_sum + 1e-12)
            h_adj = h_adj * col_factor[np.newaxis, :]
            

            h_adj = np.maximum(h_adj, 10.0)
        
        return h_adj
    
    def generate_bathymetry_with_rcont_constraints(self, row_totals, col_totals):
        nrow = len(row_totals)
        ncol = len(col_totals)
        

        if np.sum(row_totals) != np.sum(col_totals):
            raise ValueError("行总和必须等于列总和")
        if np.any(row_totals <= 0) or np.any(col_totals <= 0):
            raise ValueError("所有约束必须为正")
        
        ntotal = np.sum(row_totals)
        

        nvect = np.arange(1, ntotal + 1)
        

        nnvect = nvect.copy()
        ntemp = ntotal
        perm = np.zeros(ntotal, dtype=int)
        for i in range(ntotal):
            idx = np.random.randint(0, ntemp)
            perm[i] = nnvect[idx]
            nnvect[idx] = nnvect[ntemp - 1]
            ntemp -= 1
        

        nsubt = np.cumsum(col_totals)
        

        matrix = np.zeros((nrow, ncol), dtype=int)
        ii = 0
        for i in range(nrow):
            for k in range(row_totals[i]):
                for j in range(ncol):
                    if perm[ii] <= nsubt[j]:
                        ii += 1
                        matrix[i, j] += 1
                        break
        

        h = matrix.astype(float)

        h = h / np.max(h) * 5000.0 + 500.0
        
        return h
