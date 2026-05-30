
import numpy as np
from typing import List, Tuple, Optional
from utils import compute_triangle_area, reference_to_physical_q4, check_bounds


class MeshElement:
    def __init__(self, nodes: np.ndarray, elem_type: str = "Q4"):
        self.nodes = np.asarray(nodes, dtype=int)
        self.elem_type = elem_type
        self.level = 0
        self.load = 0.0
        self.area = 0.0


class QuadMesh:
    def __init__(self, domain: Tuple[float, float, float, float],
                 nx: int, ny: int):
        self.domain = domain
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.nx = nx
        self.ny = ny

        self.nodes = self._build_initial_nodes()
        self.elements = self._build_initial_elements()
        self._compute_element_areas()

    def _build_initial_nodes(self) -> np.ndarray:
        x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        y = np.linspace(self.ymin, self.ymax, self.ny + 1)
        nodes = []
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                nodes.append([x[i], y[j]])
        return np.array(nodes, dtype=float)

    def _build_initial_elements(self) -> List[MeshElement]:
        elements = []
        for j in range(self.ny):
            for i in range(self.nx):
                n0 = j * (self.nx + 1) + i
                n1 = n0 + 1
                n2 = n1 + (self.nx + 1)
                n3 = n0 + (self.nx + 1)
                elem = MeshElement(np.array([n0, n1, n2, n3]), "Q4")
                elements.append(elem)
        return elements

    def _compute_element_areas(self):
        for elem in self.elements:
            coords = self.nodes[elem.nodes]
            if elem.elem_type == "Q4":

                a1 = abs(compute_triangle_area(coords[0], coords[1], coords[2]))
                a2 = abs(compute_triangle_area(coords[0], coords[2], coords[3]))
                elem.area = a1 + a2
            elif elem.elem_type == "T3":
                elem.area = abs(compute_triangle_area(coords[0], coords[1], coords[2]))

    def evaluate_load(self, particles: np.ndarray) -> np.ndarray:
        particles = np.asarray(particles, dtype=float)
        loads = np.zeros(len(self.elements), dtype=int)

        for p in range(particles.shape[0]):
            x, y = particles[p]

            ix = int((x - self.xmin) / (self.xmax - self.xmin) * self.nx)
            iy = int((y - self.ymin) / (self.ymax - self.ymin) * self.ny)
            ix = max(0, min(self.nx - 1, ix))
            iy = max(0, min(self.ny - 1, iy))
            elem_idx = iy * self.nx + ix
            if elem_idx < len(self.elements):
                loads[elem_idx] += 1
        return loads

    def refine_by_load(self, particles: np.ndarray, theta: float = 0.3,
                       max_level: int = 3) -> "QuadMesh":
        loads = self.evaluate_load(particles)
        avg_load = np.mean(loads) if len(loads) > 0 else 1.0
        if avg_load < 1e-10:
            avg_load = 1.0

        threshold = avg_load * (1.0 + theta)
        new_elements = []
        new_nodes = list(self.nodes)
        node_offset = len(new_nodes)

        for idx, elem in enumerate(self.elements):
            if loads[idx] > threshold and elem.level < max_level:

                coords = self.nodes[elem.nodes]

                mid01 = 0.5 * (coords[0] + coords[1])
                mid12 = 0.5 * (coords[1] + coords[2])
                mid23 = 0.5 * (coords[2] + coords[3])
                mid30 = 0.5 * (coords[3] + coords[0])
                center = 0.25 * (coords[0] + coords[1] + coords[2] + coords[3])


                n01 = node_offset
                n12 = node_offset + 1
                n23 = node_offset + 2
                n30 = node_offset + 3
                nc = node_offset + 4
                node_offset += 5

                new_nodes.extend([mid01, mid12, mid23, mid30, center])

                n0, n1, n2, n3 = elem.nodes

                sub_elems = [
                    MeshElement(np.array([n0, n01, nc, n30]), "Q4"),
                    MeshElement(np.array([n01, n1, n12, nc]), "Q4"),
                    MeshElement(np.array([nc, n12, n2, n23]), "Q4"),
                    MeshElement(np.array([n30, nc, n23, n3]), "Q4"),
                ]
                for se in sub_elems:
                    se.level = elem.level + 1
                new_elements.extend(sub_elems)
            else:
                new_elements.append(elem)

        self.nodes = np.array(new_nodes, dtype=float)
        self.elements = new_elements
        self._compute_element_areas()
        return self

    def triangulate_elements(self) -> Tuple[np.ndarray, np.ndarray]:




        raise NotImplementedError("Hole_1: mesh_generator.py triangulate_elements 待实现")

    def get_element_centers(self) -> np.ndarray:
        centers = []
        for elem in self.elements:
            coords = self.nodes[elem.nodes]
            centers.append(np.mean(coords, axis=0))
        return np.array(centers, dtype=float)


def build_delaunay_triangulation(nodes: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(nodes)
        return tri.simplices + 1
    except ImportError:
        print("[WARNING] scipy not available; using fallback grid triangulation.")
        n = nodes.shape[0]

        nx = int(np.sqrt(n))
        ny = max(1, n // nx)
        tri_list = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                n1 = n0 + 1
                n2 = n1 + nx
                n3 = n0 + nx
                if n2 < n and n3 < n:
                    tri_list.append([n0, n1, n2])
                    tri_list.append([n0, n2, n3])
        return np.array(tri_list, dtype=int) + 1


def compute_mesh_bandwidth(element_node: np.ndarray, node_num: int) -> int:
    element_node = np.asarray(element_node, dtype=int)
    bandwidth = 0
    for e in range(element_node.shape[1]):
        nodes = element_node[:, e]
        local_bw = np.max(nodes) - np.min(nodes)
        bandwidth = max(bandwidth, local_bw)
    return bandwidth
