#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class AdaptiveMeshCover:
    
    def __init__(self):
        pass
    
    def generate_adaptive_cover(self, field, threshold=0.1, max_level=3):
        ny, nx = field.shape
        

        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)
        
        grad_x[:, :-1] = field[:, 1:] - field[:, :-1]
        grad_y[:-1, :] = field[1:, :] - field[:-1, :]
        
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        

        cover_mask = gradient > threshold
        

        for level in range(1, max_level + 1):

            cover_mask = self._dilate_mask(cover_mask)
        
        return cover_mask
    
    def _dilate_mask(self, mask):
        dilated = mask.copy()
        

        dilated[:-1, :] |= mask[1:, :]

        dilated[1:, :] |= mask[:-1, :]

        dilated[:, :-1] |= mask[:, 1:]

        dilated[:, 1:] |= mask[:, :-1]
        
        return dilated
    
    def build_cover_matrix(self, field, max_level=3):
        ny, nx = field.shape
        

        cover_info = []
        
        for level in range(max_level + 1):
            tile_size = 2 ** (max_level - level)
            
            for j in range(0, ny - tile_size + 1, tile_size):
                for i in range(0, nx - tile_size + 1, tile_size):
                    cover_info.append({
                        'level': level,
                        'size': tile_size,
                        'j': j,
                        'i': i
                    })
        
        n_vars = len(cover_info)
        n_cells = ny * nx
        

        A = np.zeros((n_cells, n_vars), dtype=int)
        b = np.ones(n_cells, dtype=int)
        
        for var_idx, info in enumerate(cover_info):
            tile_size = info['size']
            j0, i0 = info['j'], info['i']
            
            for dj in range(tile_size):
                for di in range(tile_size):
                    cell_idx = (j0 + dj) * nx + (i0 + di)
                    A[cell_idx, var_idx] = 1
        
        return A, b, cover_info
    
    def compute_cover_efficiency(self, cover_mask, max_level=3):
        ny, nx = cover_mask.shape
        total_cells = ny * nx
        

        fine_cells = total_cells
        


        coarse_cells = np.sum(~cover_mask)
        fine_cover_cells = np.sum(cover_mask)
        

        coarse_equiv = coarse_cells / (4 ** max_level)
        adaptive_equiv = coarse_equiv + fine_cover_cells
        
        savings = 1.0 - (adaptive_equiv / fine_cells)
        
        return savings
    
    def apply_cover_to_field(self, field, cover_mask, coarse_value_fn=np.mean):
        result = field.copy()
        

        ny, nx = field.shape
        block_size = 4
        
        for j in range(0, ny - block_size + 1, block_size):
            for i in range(0, nx - block_size + 1, block_size):
                block_mask = cover_mask[j:j+block_size, i:i+block_size]
                
                if not np.any(block_mask):
                    block = field[j:j+block_size, i:i+block_size]
                    coarse_val = coarse_value_fn(block)
                    result[j:j+block_size, i:i+block_size] = coarse_val
        
        return result
    
    def variomino_transformations(self, tile):
        variants = [tile.copy()]
        
        current = tile.copy()
        for _ in range(3):

            current = np.rot90(current)

            is_new = True
            for v in variants:
                if np.array_equal(current, v):
                    is_new = False
                    break
            if is_new:
                variants.append(current.copy())
        
        return variants
