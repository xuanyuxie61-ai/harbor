#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class HilbertMeshOrderer:
    
    def __init__(self, order=6):
        self.order = order
        self.N = 2 ** order
        
        if order < 1:
            raise ValueError("阶数必须 ≥ 1")
    
    def h_to_xy(self, h):

        o = h & 3
        
        if o == 0:
            x, y = 0, 0
        elif o == 1:
            x, y = 1, 0
        elif o == 2:
            x, y = 1, 1
        else:
            x, y = 0, 1
        
        h >>= 2
        w = 2
        
        while h > 0:
            o = h & 3
            xold, yold = x, y
            
            if o == 0:
                x = yold
                y = xold
            elif o == 1:
                x = xold + w
                y = yold
            elif o == 2:
                x = xold + w
                y = yold + w
            else:
                x = w - 1 - yold
                y = w - 1 - xold
            
            h >>= 2
            w *= 2
        
        return x, y
    
    def xy_to_h(self, x, y):
        h = 0
        s = self.N // 2
        

        rx, ry = 0, 0
        
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            
            h += s * s * ((3 * rx) ^ ry)
            

            if ry == 0:
                if rx == 1:
                    x = self.N - 1 - x
                    y = self.N - 1 - y
                x, y = y, x
            
            s >>= 1
        
        return h
    
    def order_2d_grid(self, nx, ny):

        max_dim = max(nx, ny)
        needed_order = int(np.ceil(np.log2(max_dim)))
        needed_order = max(needed_order, 1)
        
        N_hilbert = 2 ** needed_order
        

        indices = []
        for h in range(N_hilbert * N_hilbert):
            x, y = self._h_to_xy_with_order(h, needed_order)
            if x < nx and y < ny:
                indices.append((y, x))
        
        return np.array(indices)
    
    def _h_to_xy_with_order(self, h, order):
        o = h & 3
        
        if o == 0:
            x, y = 0, 0
        elif o == 1:
            x, y = 1, 0
        elif o == 2:
            x, y = 1, 1
        else:
            x, y = 0, 1
        
        h >>= 2
        w = 2
        
        while h > 0:
            o = h & 3
            xold, yold = x, y
            
            if o == 0:
                x = yold
                y = xold
            elif o == 1:
                x = xold + w
                y = yold
            elif o == 2:
                x = xold + w
                y = yold + w
            else:
                x = w - 1 - yold
                y = w - 1 - xold
            
            h >>= 2
            w *= 2
        
        return x, y
    
    def compute_locality_index(self, ordered_indices):
        if len(ordered_indices) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(ordered_indices) - 1):
            p1 = ordered_indices[i]
            p2 = ordered_indices[i + 1]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            total_distance += dist
        
        return total_distance / (len(ordered_indices) - 1)
    
    def compare_orderings(self, nx, ny):

        hilbert_indices = self.order_2d_grid(nx, ny)
        locality_hilbert = self.compute_locality_index(hilbert_indices)
        

        row_major = []
        for j in range(ny):
            for i in range(nx):
                row_major.append((j, i))
        row_major = np.array(row_major)
        locality_row_major = self.compute_locality_index(row_major)
        
        return locality_hilbert, locality_row_major
    
    def reorder_field(self, field, ordered_indices):
        reordered = np.zeros(len(ordered_indices))
        for idx, (j, i) in enumerate(ordered_indices):
            reordered[idx] = field[j, i]
        return reordered
    
    def inverse_reorder_field(self, reordered, ordered_indices, ny, nx):
        field = np.zeros((ny, nx))
        for idx, (j, i) in enumerate(ordered_indices):
            field[j, i] = reordered[idx]
        return field
