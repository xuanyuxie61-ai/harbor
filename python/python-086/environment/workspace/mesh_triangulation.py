# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple


class ShellTriMesh:

    def __init__(self, n_theta: int, n_x: int, geometry):
        if n_theta < 3:
            raise ValueError("环向离散份数至少为3")
        if n_x < 2:
            raise ValueError("轴向离散份数至少为2")
        self.n_theta = n_theta
        self.n_x = n_x
        self.geom = geometry
        self.nodes = None
        self.elements = None
        self.neighbors = None
        self._generate_mesh()
        self._build_neighbors()

    def _generate_mesh(self):
        R, L = self.geom.R, self.geom.L
        nt, nx = self.n_theta, self.n_x

        theta_vals = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
        x_vals = np.linspace(0.0, L, nx)
        theta_grid, x_grid = np.meshgrid(theta_vals, x_vals, indexing='ij')

        self.nodes = self.geom.parametric_surface(theta_grid, x_grid).reshape(-1, 3)
        self.n_nodes = self.nodes.shape[0]

        def node_id(i, j):
            return i * nx + j

        elements = []
        for i in range(nt):
            i_next = (i + 1) % nt
            for j in range(nx - 1):
                n1 = node_id(i, j)
                n2 = node_id(i_next, j)
                n3 = node_id(i, j + 1)
                n4 = node_id(i_next, j + 1)

                if (i + j) % 2 == 0:
                    elements.append([n1, n2, n3])
                    elements.append([n2, n4, n3])
                else:
                    elements.append([n1, n2, n4])
                    elements.append([n1, n4, n3])
        self.elements = np.array(elements, dtype=int)
        self.n_elem = self.elements.shape[0]

    def _build_neighbors(self):
        edge_map = {}
        neighbors = np.full((self.n_elem, 3), -1, dtype=int)
        for eid in range(self.n_elem):
            elem = self.elements[eid]
            edges = [(elem[1], elem[2]), (elem[2], elem[0]), (elem[0], elem[1])]
            for side, edge in enumerate(edges):
                key = tuple(sorted(edge))
                if key in edge_map:
                    other_eid, other_side = edge_map[key]
                    neighbors[eid, side] = other_eid
                    neighbors[other_eid, other_side] = eid
                else:
                    edge_map[key] = (eid, side)
        self.neighbors = neighbors

    def element_area(self, eid: int) -> float:
        nodes = self.elements[eid]
        r1, r2, r3 = self.nodes[nodes[0]], self.nodes[nodes[1]], self.nodes[nodes[2]]
        v1 = r2 - r1
        v2 = r3 - r1
        cross = np.cross(v1, v2)
        return 0.5 * np.linalg.norm(cross)

    def element_angles(self, eid: int) -> np.ndarray:
        nodes = self.elements[eid]
        r = self.nodes[nodes]
        a = np.linalg.norm(r[1] - r[2])
        b = np.linalg.norm(r[0] - r[2])
        c = np.linalg.norm(r[0] - r[1])

        eps = 1e-14
        cos_a = np.clip((b * b + c * c - a * a) / (2.0 * b * c + eps), -1.0, 1.0)
        cos_b = np.clip((a * a + c * c - b * b) / (2.0 * a * c + eps), -1.0, 1.0)
        cos_c = np.clip((a * a + b * b - c * c) / (2.0 * a * b + eps), -1.0, 1.0)
        return np.arccos(np.array([cos_a, cos_b, cos_c]))

    def alpha_measure(self) -> Tuple[float, float, float]:
        alpha_vals = []
        areas = []
        for eid in range(self.n_elem):
            angles = self.element_angles(eid)
            alpha_vals.append(np.min(angles))
            areas.append(self.element_area(eid))
        alphas = np.array(alpha_vals)
        areas = np.array(areas)
        max_angle = np.pi / 3.0
        alpha_min = np.min(alphas) / max_angle
        alpha_ave = np.mean(alphas) / max_angle
        total_area = np.sum(areas)
        if total_area > 0:
            alpha_area = np.sum(alphas * areas) / (max_angle * total_area)
        else:
            alpha_area = 0.0
        return float(alpha_min), float(alpha_ave), float(alpha_area)

    def delaunay_flip(self, max_iter: int = 10) -> int:

        flip_count = 0
        for _ in range(max_iter):
            flipped = False
            for eid in range(self.n_elem):
                for side in range(3):
                    nb = self.neighbors[eid, side]
                    if nb < 0 or nb < eid:
                        continue

                    elem = self.elements[eid]
                    nb_elem = self.elements[nb]

                    common = set(elem) & set(nb_elem)
                    if len(common) != 2:
                        continue
                    a, b = sorted(common)
                    opp_e = [n for n in elem if n not in common][0]
                    opp_nb = [n for n in nb_elem if n not in common][0]

                    p = self.nodes[[opp_e, a, opp_nb, b]]


                    def orient2d(p1, p2, p3):
                        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

                    def incircle(pa, pb, pc, pd):

                        m = np.array([
                            [pa[0] - pd[0], pa[1] - pd[1], (pa[0] - pd[0]) ** 2 + (pa[1] - pd[1]) ** 2],
                            [pb[0] - pd[0], pb[1] - pd[1], (pb[0] - pd[0]) ** 2 + (pb[1] - pd[1]) ** 2],
                            [pc[0] - pd[0], pc[1] - pd[1], (pc[0] - pd[0]) ** 2 + (pc[1] - pd[1]) ** 2]
                        ])
                        return np.linalg.det(m)

                    proj = p[:, [0, 2]]
                    if orient2d(proj[1], proj[2], proj[3]) > 0:
                        if incircle(proj[1], proj[2], proj[3], proj[0]) > 0:

                            self.elements[eid] = [opp_e, a, opp_nb]
                            self.elements[nb] = [opp_e, opp_nb, b]
                            self._build_neighbors()
                            flip_count += 1
                            flipped = True
                            break
                if flipped:
                    break
            if not flipped:
                break
        return flip_count

    def get_boundary_nodes(self) -> Tuple[np.ndarray, np.ndarray]:
        x = self.nodes[:, 2]
        tol = 1e-9
        bottom = np.where(np.abs(x) < tol)[0]
        top = np.where(np.abs(x - self.geom.L) < tol)[0]
        return bottom, top
