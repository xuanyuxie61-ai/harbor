#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import Delaunay


class FemEmbeddingInterpolator:
    
    def __init__(self, latent_dim=6):
        self.latent_dim = max(3, latent_dim)
    
    def _tetrahedron_volume(self, tetra):
        v1, v2, v3, v4 = tetra
        mat = np.array([
            v2 - v1,
            v3 - v1,
            v4 - v1
        ])
        vol = np.linalg.det(mat) / 6.0
        if abs(vol) < 1e-14:
            return 1e-14 if vol >= 0 else -1e-14
        return vol
    
    def _barycentric_coords(self, p, tetra):
        v0, v1, v2, v3 = tetra
        p = np.asarray(p, dtype=float)
        

        def signed_volume(a, b, c, d):
            mat = np.column_stack([b - a, c - a, d - a])
            return np.linalg.det(mat) / 6.0
        
        vol_total = signed_volume(v0, v1, v2, v3)
        if abs(vol_total) < 1e-14:
            return np.array([0.25, 0.25, 0.25, 0.25])
        







        lam = np.array([0.25, 0.25, 0.25, 0.25])
        return lam
    
    def _find_containing_tetrahedron(self, p, tri, points):
        p = np.asarray(p, dtype=float)
        simplex = tri.find_simplex(p)
        if simplex < 0:

            centers = tri.transform[:, :3, :3].sum(axis=1) / 3.0 + tri.transform[:, 3, :]
            dists = np.linalg.norm(centers - p, axis=1)
            simplex = np.argmin(dists)
        return simplex
    
    def interpolate(self, R_matrix):
        R = np.array(R_matrix, dtype=float)
        n_users, n_items = R.shape
        R_interp = np.copy(R)
        

        global_mean = np.nanmean(R)
        if np.isnan(global_mean):
            global_mean = 3.0
        


        R_filled = np.copy(R)
        row_means = np.nanmean(R_filled, axis=1)
        for i in range(n_users):
            if not np.isnan(row_means[i]):
                R_filled[i, np.isnan(R_filled[i, :])] = row_means[i]
            else:
                R_filled[i, :] = global_mean
        

        try:
            U, s, Vt = np.linalg.svd(R_filled - global_mean, full_matrices=False)
            coords = U[:, :3] * s[:3]
        except np.linalg.LinAlgError:
            coords = np.random.randn(n_users, 3)
        

        if n_users < 4:

            for i in range(n_users):
                R_interp[i, np.isnan(R_interp[i, :])] = row_means[i] if not np.isnan(row_means[i]) else global_mean
            return np.clip(R_interp, 1.0, 5.0)
        
        try:
            tri = Delaunay(coords)
        except Exception:

            for i in range(n_users):
                R_interp[i, np.isnan(R_interp[i, :])] = row_means[i] if not np.isnan(row_means[i]) else global_mean
            return np.clip(R_interp, 1.0, 5.0)
        

        for j in range(n_items):
            col = R[:, j]
            known = ~np.isnan(col)
            if not np.any(known):
                R_interp[:, j] = global_mean
                continue
            

            known_coords = coords[known]
            known_vals = col[known]
            
            if known_coords.shape[0] < 4:

                for i in range(n_users):
                    if np.isnan(R_interp[i, j]):
                        dists = np.linalg.norm(known_coords - coords[i], axis=1)
                        nearest = np.argmin(dists)
                        R_interp[i, j] = known_vals[nearest]
                continue
            

            try:
                local_tri = Delaunay(known_coords)
            except Exception:

                for i in range(n_users):
                    if np.isnan(R_interp[i, j]):
                        dists = np.linalg.norm(known_coords - coords[i], axis=1)
                        nearest = np.argmin(dists)
                        R_interp[i, j] = known_vals[nearest]
                continue
            

            missing = np.isnan(col)
            for i in np.where(missing)[0]:
                simplex = local_tri.find_simplex(coords[i])
                if simplex >= 0:

                    tetra = known_coords[local_tri.simplices[simplex]]
                    lam = self._barycentric_coords(coords[i], tetra)

                    lam = np.maximum(lam, 0.0)
                    lam /= lam.sum() + 1e-12
                    val = np.dot(lam, known_vals[local_tri.simplices[simplex]])
                else:

                    dists = np.linalg.norm(known_coords - coords[i], axis=1)
                    nearest = np.argmin(dists)
                    val = known_vals[nearest]
                R_interp[i, j] = val
        

        R_interp = np.clip(R_interp, 1.0, 5.0)
        return R_interp
