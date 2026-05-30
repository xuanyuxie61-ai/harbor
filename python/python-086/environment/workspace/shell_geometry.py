# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm


class CylindricalShellGeometry:

    def __init__(self, radius: float, length: float, thickness: float):
        if radius <= 0.0:
            raise ValueError("半径必须为正")
        if length <= 0.0:
            raise ValueError("长度必须为正")
        if thickness <= 0.0 or thickness >= radius:
            raise ValueError("厚度必须为正且小于半径")
        self.R = float(radius)
        self.L = float(length)
        self.t = float(thickness)

        self.K = 0.0

        self.H = 1.0 / (2.0 * self.R)

    def parametric_surface(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta)
        x = np.asarray(x)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return np.stack([
            self.R * cos_t,
            self.R * sin_t,
            x
        ], axis=-1)

    def surface_normal(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta)
        return np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)

    def first_fundamental_form(self) -> tuple:
        return self.R ** 2, 0.0, 1.0

    def second_fundamental_form(self, theta: np.ndarray) -> tuple:
        theta = np.asarray(theta)
        return self.R * np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)

    def principal_curvatures(self) -> tuple:
        return 1.0 / self.R, 0.0

    def geodesic_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        if p1.shape != (3,) or p2.shape != (3,):
            raise ValueError("输入点必须是三维坐标")

        theta1 = np.arctan2(p1[1], p1[0])
        theta2 = np.arctan2(p2[1], p2[0])
        x1, x2 = p1[2], p2[2]
        dtheta = np.abs(theta2 - theta1)
        dtheta = np.minimum(dtheta, 2.0 * np.pi - dtheta)
        dx = x2 - x1
        d = np.sqrt((self.R * dtheta) ** 2 + dx ** 2)
        return float(d)

    def boundary_sort(self, nodes: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("nodes 必须是 (N,3) 数组")

        x = nodes[:, 2]
        is_boundary = (np.abs(x) < tol) | (np.abs(x - self.L) < tol)
        indices = np.where(is_boundary)[0]
        if len(indices) == 0:
            return np.array([], dtype=int)

        theta = np.arctan2(nodes[indices, 1], nodes[indices, 0])

        theta = np.mod(theta + 2.0 * np.pi, 2.0 * np.pi)
        order = np.argsort(theta)
        return indices[order]

    def surface_area(self) -> float:
        return 2.0 * np.pi * self.R * self.L

    def aspect_ratio(self) -> float:
        return self.L / self.R

    def batdorf_parameter(self, E: float, nu: float) -> float:
        return (self.L ** 2 / (self.R * self.t)) * np.sqrt(1.0 - nu ** 2)
