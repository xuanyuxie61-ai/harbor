#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class MeshInterpolator:
    
    def __init__(self, x_coarse, y_coarse):
        self.x_coarse = x_coarse
        self.y_coarse = y_coarse
        self.nx_c = len(x_coarse)
        self.ny_c = len(y_coarse)
        self.dx_c = x_coarse[1] - x_coarse[0]
        self.dy_c = y_coarse[1] - y_coarse[0]
    
    def bilinear_interpolate(self, z_coarse, x_fine, y_fine):
        ny_f = len(y_fine)
        nx_f = len(x_fine)
        z_fine = np.zeros((ny_f, nx_f))
        
        for j in range(ny_f):
            for i in range(nx_f):
                x = x_fine[i]
                y = y_fine[j]
                

                ix = min(max(int((x - self.x_coarse[0]) / self.dx_c), 0), self.nx_c - 2)
                iy = min(max(int((y - self.y_coarse[0]) / self.dy_c), 0), self.ny_c - 2)
                

                xi = (x - self.x_coarse[ix]) / self.dx_c
                eta = (y - self.y_coarse[iy]) / self.dy_c
                

                xi = np.clip(xi, 0.0, 1.0)
                eta = np.clip(eta, 0.0, 1.0)
                

                N1 = (1.0 - xi) * (1.0 - eta)
                N2 = xi * (1.0 - eta)
                N3 = xi * eta
                N4 = (1.0 - xi) * eta
                

                z_fine[j, i] = (
                    N1 * z_coarse[iy, ix] +
                    N2 * z_coarse[iy, ix + 1] +
                    N3 * z_coarse[iy + 1, ix + 1] +
                    N4 * z_coarse[iy + 1, ix]
                )
        
        return z_fine
    
    def trigonometric_periodic_boundary(self, field, axis=0):
        field_periodic = field.copy()
        
        if axis == 0:
            n = field.shape[0]
            if n < 3:
                return field_periodic
            


            h = 1.0
            

            diff = field[0, :] - field[-1, :]
            

            smooth_width = min(3, n // 4)
            for j in range(smooth_width):
                alpha = j / smooth_width

                weight = 0.5 * (1.0 - np.cos(np.pi * alpha))
                field_periodic[j, :] = field[j, :] - diff * (1.0 - weight) * 0.5
                field_periodic[-1-j, :] = field[-1-j, :] + diff * (1.0 - weight) * 0.5
        
        elif axis == 1:
            n = field.shape[1]
            if n < 3:
                return field_periodic
            
            diff = field[:, 0] - field[:, -1]
            
            smooth_width = min(3, n // 4)
            for i in range(smooth_width):
                alpha = i / smooth_width
                weight = 0.5 * (1.0 - np.cos(np.pi * alpha))
                field_periodic[:, i] = field[:, i] - diff * (1.0 - weight) * 0.5
                field_periodic[:, -1-i] = field[:, -1-i] + diff * (1.0 - weight) * 0.5
        
        return field_periodic
    
    def cardinal_basis(self, x_nodes, x_eval, j):
        n = len(x_nodes)
        h = x_nodes[1] - x_nodes[0]
        
        if n % 2 == 1:

            if abs(x_eval - x_nodes[j]) < 1e-12:
                return 1.0
            Cj = np.sin(n * np.pi * (x_eval - x_nodes[j]) / (n * h)) / \
                 (n * np.tan(np.pi * (x_eval - x_nodes[j]) / (n * h)))
        else:

            if abs(x_eval - x_nodes[j]) < 1e-12:
                return 1.0
            Cj = np.sin(n * np.pi * (x_eval - x_nodes[j]) / (n * h)) / \
                 (n * np.sin(np.pi * (x_eval - x_nodes[j]) / (n * h)))
        
        return Cj
    
    def trigonometric_interpolate_1d(self, x_nodes, y_nodes, x_eval):
        y_eval = np.zeros_like(x_eval)
        n = len(x_nodes)
        
        for xi_idx, xi in enumerate(x_eval):
            yi = 0.0
            for j in range(n):
                yi += y_nodes[j] * self.cardinal_basis(x_nodes, xi, j)
            y_eval[xi_idx] = yi
        
        return y_eval
    
    def interpolate_quadrilateral_surface(self, nodes_xy, node_values, eval_points):
        n_eval = eval_points.shape[0]
        values = np.zeros(n_eval)
        

        x1, y1 = nodes_xy[0]
        x2, y2 = nodes_xy[1]
        x3, y3 = nodes_xy[2]
        x4, y4 = nodes_xy[3]
        
        for i in range(n_eval):
            x, y = eval_points[i]
            


            d1 = 1.0 / (np.sqrt((x - x1)**2 + (y - y1)**2) + 1e-12)
            d2 = 1.0 / (np.sqrt((x - x2)**2 + (y - y2)**2) + 1e-12)
            d3 = 1.0 / (np.sqrt((x - x3)**2 + (y - y3)**2) + 1e-12)
            d4 = 1.0 / (np.sqrt((x - x4)**2 + (y - y4)**2) + 1e-12)
            
            w_sum = d1 + d2 + d3 + d4
            w1, w2, w3, w4 = d1/w_sum, d2/w_sum, d3/w_sum, d4/w_sum
            
            values[i] = w1 * node_values[0] + w2 * node_values[1] + \
                        w3 * node_values[2] + w4 * node_values[3]
        
        return values
