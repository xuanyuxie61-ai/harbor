#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class OkadaModel:
    
    def __init__(self, strike, dip, rake, slip, length, width, depth, nu=0.25):
        self.strike = strike
        self.dip = dip
        self.rake = rake
        self.slip = slip
        self.length = length
        self.width = width
        self.depth = depth
        self.nu = nu
        

        self.shear_modulus = 30e9
        

        if not (0 <= dip <= 90):
            raise ValueError("倾角 dip 必须在 [0, 90] 范围内")
        if slip < 0:
            raise ValueError("滑动量 slip 必须非负")
        if length <= 0 or width <= 0:
            raise ValueError("断层长度和宽度必须为正")
        if depth < 0:
            raise ValueError("深度必须非负")
        if not (0 <= nu < 0.5):
            raise ValueError("泊松比必须在 [0, 0.5) 范围内")
    
    def _chinnery(self, f, x, y, L, W, d, delta):

        delta_rad = np.radians(delta)
        

        xis = [0.0, L]
        etas = [0.0, W]
        
        result = 0.0
        for xi in xis:
            for eta in etas:

                p = y * np.cos(delta_rad) + d * np.sin(delta_rad)
                q = y * np.sin(delta_rad) - d * np.cos(delta_rad)
                

                sign = 1.0
                if xi == 0:
                    sign *= -1.0
                if eta == 0:
                    sign *= -1.0
                
                result += sign * f(x, p, q, xi, eta, delta_rad)
        
        return result
    
    def _okada_vertical_displacement(self, x, y, L, W, d, delta, nu):
        delta_rad = np.radians(delta)
        sin_d = np.sin(delta_rad)
        cos_d = np.cos(delta_rad)
        

        x_prime = x
        y_prime = y * cos_d + d * sin_d
        

        zeta = y * sin_d - d * cos_d
        

        r = np.sqrt(x_prime**2 + y_prime**2 + zeta**2)
        r = np.maximum(r, 1e-6)
        


        kernel = sin_d * (
            y_prime / (r * (r + np.abs(zeta) + 1e-6))
            + (1.0 - 2.0 * nu) * np.log(r + np.abs(zeta) + 1e-6) * cos_d
        )
        
        return kernel
    
    def compute_seafloor_displacement(self, x_grid, y_grid):
        nx = len(x_grid)
        ny = len(y_grid)
        

        strike_rad = np.radians(self.strike)
        

        rake_rad = np.radians(self.rake)
        slip_strike = self.slip * np.cos(rake_rad)
        slip_dip = self.slip * np.sin(rake_rad)
        
        eta = np.zeros((ny, nx))
        
        for j in range(ny):
            for i in range(nx):

                x_obs = x_grid[i]
                y_obs = y_grid[j]
                


                x_loc = x_obs * np.cos(strike_rad) + y_obs * np.sin(strike_rad)
                y_loc = -x_obs * np.sin(strike_rad) + y_obs * np.cos(strike_rad)
                

                x_loc -= 0.0
                y_loc -= 0.0
                

                u_z_strike = slip_strike * self._strike_slip_kernel(
                    x_loc, y_loc, self.length, self.width, self.depth, self.dip, self.nu
                )
                

                u_z_dip = slip_dip * self._dip_slip_kernel(
                    x_loc, y_loc, self.length, self.width, self.depth, self.dip, self.nu
                )
                
                eta[j, i] = u_z_strike + u_z_dip
        

        eta = self._gaussian_smooth(eta, sigma=1.0)
        
        return eta
    
    def _strike_slip_kernel(self, x, y, L, W, d, delta, nu):
        delta_rad = np.radians(delta)
        sin_d = np.sin(delta_rad)
        cos_d = np.cos(delta_rad)
        


        y_prime = y * cos_d + d * sin_d
        zeta = y * sin_d - d * cos_d
        r = np.sqrt(x**2 + y_prime**2 + zeta**2)
        r = max(r, 1e-6)
        
        kernel = (sin_d / (2.0 * np.pi)) * (
            x / (r * (r + np.abs(zeta) + 1e-6))
        )
        
        return kernel
    
    def _dip_slip_kernel(self, x, y, L, W, d, delta, nu):
        delta_rad = np.radians(delta)
        sin_d = np.sin(delta_rad)
        cos_d = np.cos(delta_rad)
        
        y_prime = y * cos_d + d * sin_d
        q = y * sin_d - d * cos_d
        zeta = q
        r = np.sqrt(x**2 + y_prime**2 + zeta**2)
        r = max(r, 1e-6)
        
        kernel = (1.0 / (2.0 * np.pi)) * sin_d * (
            y_prime / (r * (r + np.abs(zeta) + 1e-6))
            + q * cos_d / (r * (r + np.abs(zeta) + 1e-6))
            + (1.0 - 2.0 * nu) * cos_d * np.log(r + np.abs(zeta) + 1e-6)
        )
        
        return kernel
    
    def _gaussian_smooth(self, field, sigma=1.0):
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=sigma)
