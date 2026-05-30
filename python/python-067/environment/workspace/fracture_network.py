# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Optional
from random_generator import MiddleSquareGenerator


class FractureNetwork:

    def __init__(self, domain_size: Tuple[float, float] = (100.0, 100.0),
                 nx: int = 50, ny: int = 50, seed: int = 1234):
        if nx <= 0 or ny <= 0:
            raise ValueError("nx 和 ny 必须为正整数")
        self.Lx, self.Ly = domain_size
        self.nx = nx
        self.ny = ny
        self.dx = self.Lx / nx
        self.dy = self.Ly / ny
        self.rng = MiddleSquareGenerator(seed=seed, d=4)


        self.aperture = np.zeros((ny, nx))

        self.connectivity = np.zeros((ny, nx), dtype=bool)

        self.orientation = np.zeros((ny, nx))

        self.transmissivity = np.zeros((ny, nx))


        self.rho = 1000.0
        self.g = 9.81
        self.mu = 1.0e-3

    def generate_ifs_fractures(self, n_iterations: int = 5000,
                                ifs_type: str = "cross") -> np.ndarray:
        if n_iterations <= 0:
            raise ValueError("n_iterations 必须为正")

        if ifs_type == "cross":

            A = np.array([[1.0/3.0, 0.0],
                          [0.0, 1.0/3.0]])
            b = np.array([
                [1.0/3.0, 0.0, 1.0/3.0, 2.0/3.0, 1.0/3.0],
                [0.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 2.0/3.0]
            ])
        elif ifs_type == "sierpinski":
            A = np.array([[0.5, 0.0],
                          [0.0, 0.5]])
            b = np.array([
                [0.0, 0.5, 0.25],
                [0.0, 0.0, 0.5]
            ])
        else:
            raise ValueError(f"不支持的 IFS 类型: {ifs_type}")

        n_maps = b.shape[1]
        x = np.zeros((2, n_iterations))
        x[:, 0] = [self.rng.random(), self.rng.random()]

        for i in range(1, n_iterations):
            j = int(self.rng.random() * n_maps) % n_maps
            x[:, i] = A @ x[:, i-1] + b[:, j]


        x[0, :] *= self.Lx
        x[1, :] *= self.Ly
        return x

    def rasterize_fractures(self, fracture_points: np.ndarray,
                            base_aperture: float = 1.0e-4,
                            aperture_std: float = 0.3) -> np.ndarray:
        if base_aperture <= 0:
            raise ValueError("base_aperture 必须为正")

        aperture = np.zeros((self.ny, self.nx))

        for i in range(fracture_points.shape[1]):
            ix = min(int(fracture_points[0, i] / self.dx), self.nx - 1)
            iy = min(int(fracture_points[1, i] / self.dy), self.ny - 1)

            log_b = np.log(base_aperture) + aperture_std * self.rng.randn()
            b_val = np.exp(log_b)
            b_val = max(b_val, 1.0e-6)
            

            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = iy + di, ix + dj
                    if 0 <= ni < self.ny and 0 <= nj < self.nx:

                        dist = np.sqrt(di**2 + dj**2)
                        factor = max(0.3, 1.0 - 0.35 * dist)
                        aperture[ni, nj] = max(aperture[ni, nj], b_val * factor)

        self.aperture = aperture
        return aperture

    def compute_transmissivity(self) -> np.ndarray:

        pass

    def update_connectivity(self, threshold: float = 1.0e-10) -> np.ndarray:
        T = self.transmissivity
        conn = T > threshold


        new_conn = conn.copy()
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if conn[i, j]:

                    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                    for ni, nj in neighbors:
                        if T[ni, nj] > threshold * 0.1:
                            new_conn[ni, nj] = True

        self.connectivity = new_conn
        return new_conn

    def check_percolation(self) -> Tuple[bool, List[Tuple[int, int]]]:
        from collections import deque

        visited = np.zeros((self.ny, self.nx), dtype=bool)
        queue = deque()
        path = {}


        for j in range(self.nx):
            if self.connectivity[0, j]:
                queue.append((0, j))
                visited[0, j] = True
                path[(0, j)] = None


        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        end_node = None

        while queue:
            i, j = queue.popleft()
            if i == self.ny - 1:
                end_node = (i, j)
                break
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.ny and 0 <= nj < self.nx:
                    if not visited[ni, nj] and self.connectivity[ni, nj]:
                        visited[ni, nj] = True
                        queue.append((ni, nj))
                        path[(ni, nj)] = (i, j)


        percolation_path = []
        if end_node is not None:
            node = end_node
            while node is not None:
                percolation_path.append(node)
                node = path[node]
            percolation_path.reverse()

        return end_node is not None, percolation_path

    def equivalent_permeability(self) -> float:
        if np.all(self.aperture == 0):
            return 0.0

        mask = self.aperture > 0
        b3_sum = np.sum(self.aperture[mask] ** 3)
        area = self.Lx * self.Ly
        k_eq = b3_sum / (12.0 * area)
        return k_eq

    def tortuosity(self, path: List[Tuple[int, int]]) -> float:
        if len(path) < 2:
            return 1.0

        actual_length = 0.0
        for k in range(len(path) - 1):
            i1, j1 = path[k]
            i2, j2 = path[k + 1]
            dx = (j2 - j1) * self.dx
            dy = (i2 - i1) * self.dy
            actual_length += np.sqrt(dx ** 2 + dy ** 2)


        euclidean_length = np.sqrt(self.Lx ** 2 + self.Ly ** 2)

        tau = actual_length / euclidean_length
        return max(tau, 1.0)

    def generate_full_network(self, n_fracture_points: int = 5000,
                              base_aperture: float = 1.0e-4) -> dict:
        points = self.generate_ifs_fractures(n_fracture_points)
        self.rasterize_fractures(points, base_aperture)
        self.compute_transmissivity()
        self.update_connectivity()
        percolates, path = self.check_percolation()
        k_eq = self.equivalent_permeability()
        tau = self.tortuosity(path) if percolates else 1.0

        return {
            "aperture": self.aperture,
            "transmissivity": self.transmissivity,
            "connectivity": self.connectivity,
            "percolates": percolates,
            "path": path,
            "equivalent_permeability": k_eq,
            "tortuosity": tau,
            "porosity": np.mean(self.connectivity),
            "domain_size": (self.Lx, self.Ly),
            "resolution": (self.nx, self.ny)
        }
