#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class HilbertLSH:
    
    def __init__(self, order=4):
        self.order = max(order, 1)
        self.grid_size = 2 ** self.order
    
    def _rmin(self, x, y, z):
        if x == 0 and y == 0 and z == 0:
            return 0
        rm = 0
        t = 1
        while t <= max(x, y, z):
            if (x & t) or (y & t) or (z & t):
                rm += 1
            t <<= 1
        return rm
    
    def xyz_to_h(self, x, y, z):
        x = int(x) & (self.grid_size - 1)
        y = int(y) & (self.grid_size - 1)
        z = int(z) & (self.grid_size - 1)
        

        rm = self._rmin(x, y, z)
        t = (self.order - rm) % 3
        
        if t == 1:
            x, y, z = z, x, y
        elif t == 2:
            x, y, z = y, z, x
        
        h = 0
        w = 2 ** (rm - 1) if rm > 0 else 2 ** (self.order - 1)
        start_k = rm if rm > 0 else self.order
        
        for k in range(start_k, 0, -1):
            xw = (x // w) & 1
            yw = (y // w) & 1
            zw = (z // w) & 1
            
            o = (xw << 2) | (yw << 1) | zw

            o_map = [
                0, 1, 3, 2,
                6, 7, 5, 4
            ]
            o = o_map[o & 7]
            

            if o == 0:
                x, y, z = y, z, x
            elif o == 1:
                x, y, z = z, x, y
            elif o == 2:
                x, y, z = z, x, y
            elif o == 3:
                x, y, z = (w - 1 - x), y, (2 * w - 1 - z)
            elif o == 4:
                x, y, z = (w - 1 - x), (y - w), (2 * w - 1 - z)
            elif o == 5:
                x, y, z = (2 * w - 1 - y), (2 * w - 1 - z), (x - w)
            elif o == 6:
                x, y, z = (2 * w - 1 - y), (w - 1 - z), (x - w)
            elif o == 7:
                x, y, z = z, (w - 1 - x), (2 * w - 1 - y)
            
            h = (h << 3) | o
            w >>= 1
        
        return h
    
    def hash_vectors(self, vectors):
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        

        vmin = vectors.min(axis=0)
        vmax = vectors.max(axis=0)
        span = vmax - vmin
        span[span < 1e-12] = 1.0
        
        normed = (vectors - vmin) / span * (self.grid_size - 1)
        normed = np.clip(normed, 0, self.grid_size - 1)
        
        hashes = np.array([
            self.xyz_to_h(int(normed[i, 0]), int(normed[i, 1]), int(normed[i, 2]))
            for i in range(vectors.shape[0])
        ])
        return hashes
    
    def approximate_nn(self, user_hashes, item_hashes, top_k=5):
        user_hashes = np.asarray(user_hashes)
        item_hashes = np.asarray(item_hashes)
        

        sorted_item_idx = np.argsort(item_hashes)
        sorted_hashes = item_hashes[sorted_item_idx]
        
        pairs = []
        for u_idx, u_h in enumerate(user_hashes):

            pos = np.searchsorted(sorted_hashes, u_h)
            

            start = max(0, pos - top_k)
            end = min(len(sorted_hashes), pos + top_k)
            neighbors = sorted_item_idx[start:end]
            
            for i_idx in neighbors[:top_k]:
                pairs.append((u_idx, i_idx))
        
        return pairs
