#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def generate_depth_grid(z_max, nz, stretch_power=2.0, z_axis=None):
    j = np.arange(nz, dtype=np.float64)
    xi = j / (nz - 1)

    eps = 1e-10
    xi = np.clip(xi, eps, 1.0 - eps)
    z = z_max * (xi ** stretch_power) / (xi ** stretch_power + (1.0 - xi) ** stretch_power)

    if z_axis is not None and 0 < z_axis < z_max:

        sigma = z_max * 0.15
        compression = 1.0 - 0.3 * np.exp(-0.5 * ((z - z_axis) / sigma) ** 2)
        z = np.cumsum(np.diff(np.concatenate([[0], z])) * compression)
        z = z_max * z / z[-1]
    z[0] = 0.0
    z[-1] = z_max
    return z


def generate_range_grid(r_max, dr):
    n_r = int(np.ceil(r_max / dr)) + 1
    return np.linspace(0.0, r_max, n_r)


def point_in_polygon(x_poly, y_poly, x0, y0):
    x_poly = np.asarray(x_poly, dtype=np.float64)
    y_poly = np.asarray(y_poly, dtype=np.float64)
    n = len(x_poly)
    if n < 3:
        return False
    inside = False
    x1 = x_poly[-1]
    y1 = y_poly[-1]
    for i in range(n):
        x2 = x_poly[i]
        y2 = y_poly[i]

        if (y1 > y0) != (y2 > y0):
            t = (x2 - x1) * (y0 - y1) / (y2 - y1 + 1e-15) + x1
            if t > x0:
                inside = not inside
        x1, y1 = x2, y2
    return inside


class PEMesh:

    def __init__(self, r_grid, z_grid, env):
        self.r_grid = np.asarray(r_grid, dtype=np.float64)
        self.z_grid = np.asarray(z_grid, dtype=np.float64)
        self.env = env
        self.nr = len(r_grid)
        self.nz = len(z_grid)
        self.dr = r_grid[1] - r_grid[0] if self.nr > 1 else 1.0
        self.dz = np.diff(z_grid)
        self.dz = np.concatenate([self.dz, [self.dz[-1]]])

        self.R, self.Z = np.meshgrid(self.r_grid, self.z_grid, indexing='ij')

        self.seafloor_depth = self.env.bathymetry(self.r_grid)

        self.node_mask = np.zeros((self.nr, self.nz), dtype=bool)
        for m in range(self.nr):
            h_b = self.seafloor_depth[m]
            self.node_mask[m, :] = self.z_grid <= h_b + 1e-6

        self.node_mask[:, 0] = True

        self.num_valid_nodes = np.sum(self.node_mask)

        self.elements = self._build_triangular_elements()

    def _build_triangular_elements(self):
        elements = []
        for m in range(self.nr - 1):
            for n in range(self.nz - 1):

                i1 = m * self.nz + n
                i2 = (m + 1) * self.nz + n
                i3 = (m + 1) * self.nz + (n + 1)
                i4 = m * self.nz + (n + 1)

                mask1 = self.node_mask[m, n]
                mask2 = self.node_mask[m + 1, n]
                mask3 = self.node_mask[m + 1, n + 1]
                mask4 = self.node_mask[m, n + 1]
                if mask1 and mask2 and mask4:
                    elements.append([i1, i2, i4])
                if mask2 and mask3 and mask4:
                    elements.append([i2, i3, i4])
        return np.asarray(elements, dtype=np.int64)

    def get_1d_slice(self, m):
        return self.z_grid.copy(), self.node_mask[m, :].copy()

    def global_index(self, m, n):
        return m * self.nz + n

    def local_index(self, idx):
        m = idx // self.nz
        n = idx % self.nz
        return m, n

    def adaptive_range_step(self, m, safety_factor=0.5):
        z_valid = self.z_grid[self.node_mask[m, :]]
        if len(z_valid) == 0:
            return self.dr
        n2_dev = self.env.refractive_index_squared_deviation(z_valid)
        denom = self.env.k0 * np.abs(n2_dev)
        denom = np.maximum(denom, 1e-12)
        dr_adaptive = safety_factor * 2.0 / np.max(denom)
        return min(dr_adaptive, self.dr * 2.0)

    def mesh_quality_stats(self):
        if len(self.elements) == 0:
            return {}

        ar_list = []
        for elem in self.elements:
            nodes = []
            for idx in elem:
                m, n = self.local_index(idx)
                nodes.append((self.R[m, n], self.Z[m, n]))
            nodes = np.asarray(nodes)

            d = [np.linalg.norm(nodes[i] - nodes[(i + 1) % 3]) for i in range(3)]
            if min(d) > 1e-9:
                ar_list.append(max(d) / min(d))
        return {
            'num_elements': len(self.elements),
            'num_valid_nodes': int(self.num_valid_nodes),
            'aspect_ratio_mean': float(np.mean(ar_list)) if ar_list else 0.0,
            'aspect_ratio_max': float(np.max(ar_list)) if ar_list else 0.0,
        }
