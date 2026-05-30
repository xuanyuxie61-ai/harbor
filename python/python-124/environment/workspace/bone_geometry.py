
import numpy as np
from typing import Tuple, List, Optional


class BoneGeometry:

    def __init__(self, width: float = 20.0, height: float = 30.0,
                 cortical_thickness: float = 2.0, nx: int = 17, ny: int = 17):
        if nx % 2 == 0 or ny % 2 == 0:
            raise ValueError("nx and ny must be odd for quadratic elements.")
        if width <= 0 or height <= 0 or cortical_thickness <= 0:
            raise ValueError("Geometric dimensions must be positive.")
        if cortical_thickness >= min(width, height) / 2.0:
            raise ValueError("Cortical thickness too large for given bone dimensions.")

        self.width = width
        self.height = height
        self.cortical_thickness = cortical_thickness
        self.nx = nx
        self.ny = ny
        self.element_order = 6


        self.element_num = (nx - 1) * (ny - 1) * 2
        self.node_num = (2 * nx - 1) * (2 * ny - 1)


        self.node_xy = self._generate_nodes()
        self.element_node = self._generate_t6_grid()
        self.element_area = self._compute_element_areas()
        self.triangle_neighbors = self._compute_triangle_neighbors()
        self.node_distances = self._compute_node_boundary_distances()




    def _generate_nodes(self) -> np.ndarray:
        nx, ny = self.nx, self.ny
        node_num = self.node_num
        node_xy = np.zeros((2, node_num))


        dx = self.width / (nx - 1)
        dy = self.height / (ny - 1)


        nx2 = 2 * nx - 1
        ny2 = 2 * ny - 1
        dx2 = self.width / (nx2 - 1)
        dy2 = self.height / (ny2 - 1)

        node = 0
        for j in range(ny2):
            for i in range(nx2):
                node_xy[0, node] = i * dx2
                node_xy[1, node] = j * dy2
                node += 1

        if node != node_num:
            raise RuntimeError(f"Node count mismatch: {node} != {node_num}")
        return node_xy




    def _generate_t6_grid(self) -> np.ndarray:
        nx, ny = self.nx, self.ny
        element_num = self.element_num
        element_order = self.element_order
        element_node = np.zeros((element_order, element_num), dtype=int)


        nx2 = 2 * nx - 1

        element = 0
        for j in range(ny - 1):
            for i in range(nx - 1):


                sw = 2 * j * nx2 + 2 * i
                se = sw + 2
                nw = sw + 2 * nx2
                ne = nw + 2
                s_mid = sw + 1
                e_mid = se + nx2
                n_mid = nw + 1
                w_mid = sw + nx2
                c_mid = sw + nx2 + 1


                element_node[0, element] = sw
                element_node[1, element] = se
                element_node[2, element] = nw
                element_node[3, element] = s_mid
                element_node[4, element] = c_mid
                element_node[5, element] = w_mid
                element += 1


                element_node[0, element] = ne
                element_node[1, element] = nw
                element_node[2, element] = se
                element_node[3, element] = n_mid
                element_node[4, element] = c_mid
                element_node[5, element] = e_mid
                element += 1

        if element != element_num:
            raise RuntimeError(f"Element count mismatch: {element} != {element_num}")
        return element_node




    def _compute_element_areas(self) -> np.ndarray:
        element_num = self.element_num
        element_area = np.zeros(element_num)

        for e in range(element_num):
            i1 = self.element_node[0, e]
            i2 = self.element_node[1, e]
            i3 = self.element_node[2, e]

            x1, y1 = self.node_xy[:, i1]
            x2, y2 = self.node_xy[:, i2]
            x3, y3 = self.node_xy[:, i3]

            area = 0.5 * abs(
                y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)
            )
            if area < 1e-14:
                raise ValueError(f"Degenerate triangle element {e} with area {area}")
            element_area[e] = area

        return element_area




    def _compute_triangle_neighbors(self) -> np.ndarray:
        element_num = self.element_num
        element_order = self.element_order
        neighbors = np.full((3, element_num), -1, dtype=int)


        edge_list = []
        for e in range(element_num):

            n1 = self.element_node[0, e]
            n2 = self.element_node[1, e]
            n3 = self.element_node[2, e]
            edges = [(n1, n2), (n2, n3), (n3, n1)]
            for side, (a, b) in enumerate(edges):
                edge_list.append((min(a, b), max(a, b), side, e))


        edge_list.sort(key=lambda t: (t[0], t[1]))


        i = 0
        while i < len(edge_list):
            j = i + 1
            if j < len(edge_list) and edge_list[i][0] == edge_list[j][0] and \
                    edge_list[i][1] == edge_list[j][1]:
                side1, elem1 = edge_list[i][2], edge_list[i][3]
                side2, elem2 = edge_list[j][2], edge_list[j][3]
                neighbors[side1, elem1] = elem2
                neighbors[side2, elem2] = elem1
                i += 2
            else:
                i += 1

        return neighbors




    def _compute_node_boundary_distances(self) -> np.ndarray:
        node_num = self.node_num
        distances = np.zeros(node_num)

        w, h = self.width, self.height

        for node in range(node_num):
            x, y = self.node_xy[:, node]


            d_left = x
            d_right = w - x
            d_bottom = y
            d_top = h - y


            distances[node] = min(d_left, d_right, d_bottom, d_top)

        return distances

    def is_cortical(self, node_index: int) -> bool:
        return self.node_distances[node_index] <= self.cortical_thickness

    def classify_nodes(self) -> Tuple[np.ndarray, np.ndarray]:
        cortical = np.array([i for i in range(self.node_num)
                             if self.is_cortical(i)], dtype=int)
        trabecular = np.array([i for i in range(self.node_num)
                               if not self.is_cortical(i)], dtype=int)
        return cortical, trabecular

    def export_nodes_elements(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.node_xy.copy(), self.element_node.copy() + 1

    def compute_half_bandwidth(self) -> int:
        nhba = 0
        for e in range(self.element_num):
            for iln in range(self.element_order):
                i = self.element_node[iln, e]
                if i >= 0:
                    for jln in range(self.element_order):
                        j = self.element_node[jln, e]
                        nhba = max(nhba, abs(j - i))
        return nhba





def point_line_distance_signed(p1: np.ndarray, p2: np.ndarray,
                               p: np.ndarray) -> float:
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p = np.asarray(p, dtype=float)

    l_dv = p2 - p1
    norm_dv = np.linalg.norm(l_dv)
    if norm_dv < 1e-14:
        raise ValueError("p1 and p2 must be distinct.")

    l_nv = np.array([-l_dv[1], l_dv[0]])
    l_nv = l_nv / np.linalg.norm(l_nv)

    dist = float(np.dot(l_nv, p - p1))
    return dist
