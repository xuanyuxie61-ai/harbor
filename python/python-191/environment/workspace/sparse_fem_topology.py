
import numpy as np
from typing import Tuple, List, Dict
import math


class TriangularMesh:
    
    def __init__(self, nodes: np.ndarray = None, elements: np.ndarray = None):
        if nodes is not None:
            nodes = np.asarray(nodes, dtype=np.float64)
            if nodes.shape[0] == 3 and nodes.shape[1] != 3:
                nodes = nodes.T
            self.nodes = nodes
        else:
            self.nodes = np.zeros((0, 3))
        
        if elements is not None:
            elements = np.asarray(elements, dtype=np.int64)
            if elements.shape[0] == 3 and elements.shape[1] != 3:
                elements = elements.T
            self.elements = elements
        else:
            self.elements = np.zeros((0, 3), dtype=np.int64)
        
        self._validate()
    
    def _validate(self):
        if self.nodes.size == 0 or self.elements.size == 0:
            return
        
        n_nodes = self.nodes.shape[0]
        if self.nodes.shape[1] != 3:
            raise ValueError(f"Nodes must have 3 coordinates, got {self.nodes.shape[1]}")
        if self.elements.shape[1] != 3:
            raise ValueError(f"Elements must have 3 nodes, got {self.elements.shape[1]}")
        
        max_idx = np.max(self.elements)
        min_idx = np.min(self.elements)
        if min_idx < 0 or max_idx >= n_nodes:
            raise ValueError(
                f"Element indices out of range [0, {n_nodes-1}], got [{min_idx}, {max_idx}]"
            )
    
    def node_degrees(self) -> np.ndarray:
        n_nodes = self.nodes.shape[0]
        degrees = np.zeros(n_nodes, dtype=np.int64)
        for tri in self.elements:
            for v in tri:
                degrees[v] += 1
        return degrees
    
    def triangle_areas(self) -> np.ndarray:
        if self.elements.shape[0] == 0:
            return np.array([])
        
        v1 = self.nodes[self.elements[:, 0], :]
        v2 = self.nodes[self.elements[:, 1], :]
        v3 = self.nodes[self.elements[:, 2], :]
        
        cross = np.cross(v2 - v1, v3 - v1)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        return areas
    
    def build_sparse_pattern(self) -> Dict:
        n_nodes = self.nodes.shape[0]
        adj = [set() for _ in range(n_nodes)]
        
        for tri in self.elements:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        adj[tri[i]].add(tri[j])
        
        indptr = [0]
        indices = []
        for i in range(n_nodes):
            neighbors = sorted(adj[i])
            indices.extend(neighbors)
            indptr.append(len(indices))
        
        return {
            'n_rows': n_nodes,
            'n_cols': n_nodes,
            'indptr': np.array(indptr, dtype=np.int64),
            'indices': np.array(indices, dtype=np.int64),
            'nnz': len(indices)
        }
    
    def assemble_stiffness_matrix_2d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.elements.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
        
        data_list = []
        row_list = []
        col_list = []
        
        for tri in self.elements:
            v = self.nodes[tri, :2]
            

            area = 0.5 * abs(
                v[0, 0] * (v[1, 1] - v[2, 1]) +
                v[1, 0] * (v[2, 1] - v[0, 1]) +
                v[2, 0] * (v[0, 1] - v[1, 1])
            )
            
            if area < 1e-14:
                continue
            

            grad = np.zeros((3, 2))
            grad[0, 0] = v[1, 1] - v[2, 1]
            grad[0, 1] = v[2, 0] - v[1, 0]
            grad[1, 0] = v[2, 1] - v[0, 1]
            grad[1, 1] = v[0, 0] - v[2, 0]
            grad[2, 0] = v[0, 1] - v[1, 1]
            grad[2, 1] = v[1, 0] - v[0, 0]
            grad /= (2.0 * area)
            

            for i in range(3):
                for j in range(3):
                    val = area * np.dot(grad[i], grad[j])
                    data_list.append(val)
                    row_list.append(tri[i])
                    col_list.append(tri[j])
        
        return (
            np.array(data_list, dtype=np.float64),
            np.array(row_list, dtype=np.int64),
            np.array(col_list, dtype=np.int64)
        )


def trinity_tile_cover_pattern(
    region_triangles: int,
    tile_types: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    if region_triangles < 1 or tile_types < 1:
        raise ValueError("region_triangles and tile_types must be >= 1")
    

    configs_per_tile = 3
    total_vars = tile_types * configs_per_tile
    
    A1 = np.zeros((region_triangles, total_vars), dtype=np.int32)
    A2 = np.zeros((tile_types, total_vars), dtype=np.int32)
    
    np.random.seed(42)
    for t in range(tile_types):
        for c in range(configs_per_tile):
            var_idx = t * configs_per_tile + c

            A2[t, var_idx] = 1

            n_cover = min(3, region_triangles)
            cover_idx = np.random.choice(region_triangles, size=n_cover, replace=False)
            A1[cover_idx, var_idx] = 1
    
    return A1, A2


def sparse_matrix_vector_product(
    data: np.ndarray,
    row_ind: np.ndarray,
    col_ind: np.ndarray,
    x: np.ndarray,
    n_rows: int
) -> np.ndarray:
    y = np.zeros(n_rows, dtype=np.float64)
    for val, i, j in zip(data, row_ind, col_ind):
        if 0 <= i < n_rows and 0 <= j < len(x):
            y[i] += val * x[j]
    return y


def sparsity_ratio(data: np.ndarray, n_rows: int, n_cols: int) -> float:
    if n_rows == 0 or n_cols == 0:
        return 0.0
    nnz = len(data)
    return 1.0 - nnz / (n_rows * n_cols)


if __name__ == "__main__":

    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 0.0]
    ])
    elements = np.array([
        [0, 1, 2],
        [0, 3, 1]
    ])
    mesh = TriangularMesh(nodes, elements)
    print("Areas:", mesh.triangle_areas())
    print("Degrees:", mesh.node_degrees())
    
    data, rows, cols = mesh.assemble_stiffness_matrix_2d()
    print("Stiffness nnz:", len(data))
    
    A1, A2 = trinity_tile_cover_pattern(6, 3)
    print("Trinity A1 shape:", A1.shape, "A2 shape:", A2.shape)
