
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class FEM2DSolver:

    def __init__(self, nodes, elements, conductivity=0.3):
        nodes = np.asarray(nodes, dtype=float)
        elements = np.asarray(elements, dtype=int)

        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes must be of shape (N, 2)")
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError("elements must be of shape (M, 3)")
        if np.any(elements < 0) or np.any(elements >= nodes.shape[0]):
            raise ValueError("elements 包含越界节点索引")

        self.nodes = nodes
        self.elements = elements
        self.n_nodes = nodes.shape[0]
        self.n_elements = elements.shape[0]

        if np.isscalar(conductivity):
            self.sigma = np.full(self.n_elements, float(conductivity))
        else:
            self.sigma = np.asarray(conductivity, dtype=float)
            if self.sigma.shape != (self.n_elements,):
                raise ValueError("conductivity 长度必须与单元数相同")

        self._stiffness = None
        self._built = False

    def _build_stiffness_matrix(self):
        row_idx = []
        col_idx = []
        data = []

        for elem_idx, elem in enumerate(self.elements):
            pts = self.nodes[elem]

            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            if area < 1e-14:
                continue




            b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2.0 * area)
            c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2.0 * area)


            sigma = self.sigma[elem_idx]
            ke = sigma * area * (np.outer(b, b) + np.outer(c, c))

            for i in range(3):
                for j in range(3):
                    row_idx.append(elem[i])
                    col_idx.append(elem[j])
                    data.append(ke[i, j])

        K = csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(self.n_nodes, self.n_nodes)
        )
        self._stiffness = K
        self._built = True

    def solve(self, source, dirichlet_nodes=None, dirichlet_values=None):
        source = np.asarray(source, dtype=float)
        if source.shape != (self.n_nodes,):
            raise ValueError("source 长度必须与节点数相同")

        if not self._built:
            self._build_stiffness_matrix()

        K = self._stiffness.copy()
        F = source.copy()


        if dirichlet_nodes is not None and dirichlet_values is not None:
            dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=int)
            dirichlet_values = np.asarray(dirichlet_values, dtype=float)
            for idx, val in zip(dirichlet_nodes, dirichlet_values):

                F -= K[:, idx].toarray().flatten() * val
                F[idx] = val

                K[idx, :] = 0.0
                K[:, idx] = 0.0
                K[idx, idx] = 1.0


        V = spsolve(K, F)
        if V is None:
            raise RuntimeError("线性系统求解失败，刚度矩阵可能奇异")
        return V

    def compute_gradient(self, V):
        V = np.asarray(V, dtype=float)
        grad = np.zeros((self.n_elements, 2))
        for elem_idx, elem in enumerate(self.elements):
            pts = self.nodes[elem]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            if area < 1e-14:
                continue
            b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2.0 * area)
            c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2.0 * area)
            v_elem = V[elem]
            grad[elem_idx, 0] = np.dot(b, v_elem)
            grad[elem_idx, 1] = np.dot(c, v_elem)
        return grad

    def compute_activation_function(self, V):

        grad = self.compute_gradient(V)

        node_grad = np.zeros((self.n_nodes, 2))
        node_count = np.zeros(self.n_nodes)
        for elem_idx, elem in enumerate(self.elements):
            for ni in elem:
                node_grad[ni] += grad[elem_idx]
                node_count[ni] += 1
        node_count = np.where(node_count < 1, 1, node_count)
        node_grad /= node_count[:, np.newaxis]









        af = np.zeros(self.n_nodes)
        raise NotImplementedError("Hole 1: 请实现 compute_activation_function 的最小二乘拟合")

        return af


def generate_cochlea_mesh(geometry, n_radial=20, n_angular=80):
    theta = np.linspace(0.0, geometry.theta_max, n_angular)
    nodes_list = []
    node_indices = np.empty((n_angular, n_radial), dtype=int)
    idx = 0
    for ti, th in enumerate(theta):
        center = geometry.centerline_at(th)
        r = geometry.r0 * np.exp(-geometry.b * th)

        dr = -geometry.b * r
        dx = dr * np.cos(th) - r * np.sin(th)
        dy = dr * np.sin(th) + r * np.cos(th)
        tangent = np.array([dx, dy])
        tangent = tangent / (np.linalg.norm(tangent) + 1e-14)
        normal = np.array([-tangent[1], tangent[0]])


        for ri in range(n_radial):
            frac = ri / (n_radial - 1)
            offset = -geometry.scala_width / 2 + frac * geometry.scala_width
            pos = center + offset * normal
            nodes_list.append(pos)
            node_indices[ti, ri] = idx
            idx += 1

    nodes = np.array(nodes_list)


    elements = []
    for ti in range(n_angular - 1):
        for ri in range(n_radial - 1):
            n1 = node_indices[ti, ri]
            n2 = node_indices[ti + 1, ri]
            n3 = node_indices[ti, ri + 1]
            n4 = node_indices[ti + 1, ri + 1]
            elements.append([n1, n2, n3])
            elements.append([n2, n4, n3])

    elements = np.array(elements, dtype=int)
    return nodes, elements
