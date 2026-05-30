# -*- coding: utf-8 -*-

import numpy as np
from math import sin, cos, pi, sqrt, atan2


class BoundaryLayerMesh:

    def __init__(self, L=1.0, H=0.1, Nx=100, Ny=80, Re=1e6, Ma=6.0):
        self.L = L
        self.H = H
        self.Nx = Nx
        self.Ny = Ny
        self.Re = Re
        self.Ma = Ma

    def wall_normal_stretching(self, n, h_max, beta=1.08):



        Re_L = self.Re
        delta = 5.0 * self.L / sqrt(Re_L)


        rho_e = 1.225
        a = 340.0
        u_e = self.Ma * a
        cf = 0.664 / sqrt(Re_L)
        tau_w = 0.5 * rho_e * u_e**2 * cf


        mu_w = 1.7894e-5 * 1.458e-6
        rho_w = rho_e
        nu_w = mu_w / rho_w
        u_tau = sqrt(max(tau_w / rho_w, 1e-20))


        dy1 = max(nu_w / u_tau, 1e-8)




        beta_est = beta
        for _ in range(20):
            denom = beta_est - 1.0
            if abs(denom) < 1e-12:
                break
            h_est = dy1 * (beta_est**n - 1.0) / denom
            if abs(h_est - h_max) < 1e-8:
                break

            df = dy1 * (n * beta_est**(n - 1) * denom - (beta_est**n - 1.0)) / (denom**2)
            if abs(df) > 1e-12:
                beta_est = max(1.001, beta_est - (h_est - h_max) / df)

        y = np.zeros(n + 1)
        for j in range(1, n + 1):
            y[j] = dy1 * (beta_est**j - 1.0) / (beta_est - 1.0)
        y = np.clip(y, 0.0, h_max)
        y[-1] = h_max
        return y

    def generate_flat_plate_mesh(self):
        x = np.linspace(0.0, self.L, self.Nx)
        y = self.wall_normal_stretching(self.Ny - 1, self.H)
        nx, ny = len(x), len(y)

        nodes = np.zeros((nx * ny, 2))
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                nodes[idx, 0] = x[i]
                nodes[idx, 1] = y[j]
        return nodes, nx, ny

    def generate_triangles_from_structured(self, nx, ny):
        n_tri = (nx - 1) * (ny - 1) * 2
        triangles = np.zeros((n_tri, 3), dtype=int)
        t = 0
        for i in range(nx - 1):
            for j in range(ny - 1):
                n1 = i * ny + j
                n2 = (i + 1) * ny + j
                n3 = (i + 1) * ny + (j + 1)
                n4 = i * ny + (j + 1)
                triangles[t] = [n1, n2, n3]
                triangles[t + 1] = [n1, n3, n4]
                t += 2
        return triangles

    def triangle_neighbors(self, triangle_num, triangle_node):
        tn = triangle_node
        if tn.shape[0] == 3 and tn.shape[1] == triangle_num:
            tn = tn.T

        neighbors = np.full((triangle_num, 3), -1, dtype=int)


        edge_map = {}
        for t in range(triangle_num):
            for s in range(3):
                n1 = tn[t, s]
                n2 = tn[t, (s + 1) % 3]
                key = (min(n1, n2), max(n1, n2))
                if key not in edge_map:
                    edge_map[key] = []
                edge_map[key].append((t, s))


        for t in range(triangle_num):
            for s in range(3):
                n1 = tn[t, s]
                n2 = tn[t, (s + 1) % 3]
                key = (min(n1, n2), max(n1, n2))
                candidates = edge_map[key]
                for (t2, s2) in candidates:
                    if t2 != t:

                        n1b = tn[t2, s2]
                        n2b = tn[t2, (s2 + 1) % 3]
                        if n1 == n2b and n2 == n1b:
                            neighbors[t, s] = t2
                            break
        return neighbors

    def boundary_nodes(self, nx, ny):
        wall = [i * ny for i in range(nx)]
        inlet = [j for j in range(ny)]
        outlet = [(nx - 1) * ny + j for j in range(ny)]
        farfield = [i * ny + (ny - 1) for i in range(nx)]
        return {
            'wall': np.array(wall, dtype=int),
            'inlet': np.array(inlet, dtype=int),
            'outlet': np.array(outlet, dtype=int),
            'farfield': np.array(farfield, dtype=int)
        }


def sphere_wavevector_grid(lat_num, long_num):


    j = np.arange(lat_num + 2)
    phi_nodes = np.pi * j / (lat_num + 1)

    point_num = 2 + lat_num * long_num
    p = np.zeros((point_num, 3))
    n = 0


    p[n] = [0.0, 0.0, 1.0]
    n += 1

    for lat in range(1, lat_num + 1):
        phi = phi_nodes[lat]
        for lng in range(long_num):
            theta = 2.0 * pi * lng / long_num
            p[n, 0] = sin(phi) * cos(theta)
            p[n, 1] = sin(phi) * sin(theta)
            p[n, 2] = cos(phi)
            n += 1


    p[n] = [0.0, 0.0, -1.0]
    n += 1
    return p


def save_xy_data(filename, x, y):
    data = np.column_stack((x, y))
    np.savetxt(filename, data, fmt='%.8e', header='X Y', comments='# ')


def read_xy_data(filename):
    data = np.loadtxt(filename, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1]
