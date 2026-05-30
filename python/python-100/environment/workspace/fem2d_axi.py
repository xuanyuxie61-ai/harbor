
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class FEM2DAxi:

    MU0 = 4.0 * np.pi * 1.0e-7

    def __init__(self, mesh):
        from mesh_engine import Mesh2D

        if not isinstance(mesh, Mesh2D):
            raise TypeError("mesh 必须是 Mesh2D 实例")
        self.mesh = mesh
        self.n_dof = mesh.n_nodes()

    def _shape_gradients(self, elem_idx: int) -> tuple:
        v = self.mesh.elements[elem_idx]
        p1 = self.mesh.nodes[v[0]]
        p2 = self.mesh.nodes[v[1]]
        p3 = self.mesh.nodes[v[2]]

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if abs(area) < 1.0e-16:
            raise ValueError(f"退化单元 {elem_idx}: 面积接近零")


        grads = np.zeros((3, 2))
        grads[0, 0] = (y2 - y3) / (2.0 * area)
        grads[0, 1] = (x3 - x2) / (2.0 * area)
        grads[1, 0] = (y3 - y1) / (2.0 * area)
        grads[1, 1] = (x1 - x3) / (2.0 * area)
        grads[2, 0] = (y1 - y2) / (2.0 * area)
        grads[2, 1] = (x2 - x1) / (2.0 * area)

        return grads, area

    def assemble_linear(self, nu_func, source_func, elem_tags_filter: set = None) -> tuple:
        n_dof = self.n_dof
        row_idx = []
        col_idx = []
        data = []
        F = np.zeros(n_dof)

        for e in range(self.mesh.n_elements()):
            if elem_tags_filter is not None and self.mesh.elem_tags[e] not in elem_tags_filter:
                continue

            v = self.mesh.elements[e]
            grads, area = self._shape_gradients(e)


            centroid = np.mean(self.mesh.nodes[v], axis=0)
            nu_val = nu_func(centroid[0], centroid[1])


            raise NotImplementedError("Hole_2: 需实现2D FEM刚度矩阵组装与载荷向量计算")

        K = csr_matrix((data, (row_idx, col_idx)), shape=(n_dof, n_dof))
        return K, F

    def apply_dirichlet(self, K: csr_matrix, F: np.ndarray, bc_nodes: dict) -> tuple:
        K = K.tolil()
        for idx, val in bc_nodes.items():
            if idx < 0 or idx >= self.n_dof:
                raise ValueError(f"Dirichlet节点索引越界: {idx}")
            penalty = 1.0e16
            K[idx, idx] = penalty
            F[idx] = penalty * val
        return K.tocsr(), F

    def solve_linear(self, K: csr_matrix, F: np.ndarray) -> np.ndarray:

        diag = K.diagonal()
        zero_diag = np.abs(diag) < 1.0e-20
        if np.any(zero_diag):
            K = K.tolil()
            for i in np.where(zero_diag)[0]:
                K[i, i] = 1.0e-12
            K = K.tocsr()

        A = spsolve(K, F)
        if A is None:
            raise RuntimeError("线性系统求解失败，矩阵可能奇异")
        return np.asarray(A)

    def compute_b_field_at_nodes(self, A: np.ndarray) -> tuple:
        n_dof = self.n_dof
        Bx = np.zeros(n_dof)
        By = np.zeros(n_dof)
        weight = np.zeros(n_dof)

        for e in range(self.mesh.n_elements()):
            v = self.mesh.elements[e]
            grads, area = self._shape_gradients(e)


            dAdx = 0.0
            dAdy = 0.0
            for j in range(3):
                dAdx += A[v[j]] * grads[j, 0]
                dAdy += A[v[j]] * grads[j, 1]

            Bx_elem = dAdy
            By_elem = -dAdx

            for j in range(3):
                vj = v[j]
                Bx[vj] += area * Bx_elem
                By[vj] += area * By_elem
                weight[vj] += area


        safe_w = np.where(weight < 1.0e-14, 1.0, weight)
        Bx /= safe_w
        By /= safe_w
        return Bx, By

    def compute_maxwell_stress_tensor(self, Bx: np.ndarray, By: np.ndarray) -> tuple:
        B2 = Bx * Bx + By * By
        factor = 1.0 / self.MU0
        Txx = factor * (Bx * Bx - 0.5 * B2)
        Txy = factor * (Bx * By)
        Tyy = factor * (By * By - 0.5 * B2)
        return Txx, Txy, Tyy

    def compute_electromagnetic_torque(
        self, A: np.ndarray, radius_airgap_inner: float, radius_airgap_outer: float
    ) -> float:
        Bx, By = self.compute_b_field_at_nodes(A)
        Txx, Txy, Tyy = self.compute_maxwell_stress_tensor(Bx, By)


        r_mid = 0.5 * (radius_airgap_inner + radius_airgap_outer)
        tol = 0.5 * (radius_airgap_outer - radius_airgap_inner)

        torque = 0.0
        count = 0
        L = 1.0

        for i in range(self.n_dof):
            x, y = self.mesh.nodes[i]
            r = np.sqrt(x * x + y * y)
            if abs(r - r_mid) < tol and r > 1.0e-10:
                theta = np.arctan2(y, x)



                Br = Bx[i] * np.cos(theta) + By[i] * np.sin(theta)
                Bt = -Bx[i] * np.sin(theta) + By[i] * np.cos(theta)
                Trt = (1.0 / self.MU0) * Br * Bt
                torque += r * r * Trt
                count += 1

        if count > 0:

            torque = torque / count * 2.0 * np.pi * L
        return float(torque)

    def compute_magnetic_energy(self, A: np.ndarray, nu_func) -> float:
        W = 0.0
        for e in range(self.mesh.n_elements()):
            v = self.mesh.elements[e]
            grads, area = self._shape_gradients(e)
            centroid = np.mean(self.mesh.nodes[v], axis=0)
            nu_val = nu_func(centroid[0], centroid[1])

            dAdx = sum(A[v[j]] * grads[j, 0] for j in range(3))
            dAdy = sum(A[v[j]] * grads[j, 1] for j in range(3))
            B2 = dAdx * dAdx + dAdy * dAdy
            W += 0.5 * nu_val * B2 * area

        return W

    def compute_eddy_current_loss(
        self, A: np.ndarray, sigma: float, omega: float, elem_tags_filter: set = None
    ) -> float:
        P = 0.0
        coeff = 0.5 * omega * omega * sigma
        for e in range(self.mesh.n_elements()):
            if elem_tags_filter is not None and self.mesh.elem_tags[e] not in elem_tags_filter:
                continue

            v = self.mesh.elements[e]
            _, area = self._shape_gradients(e)
            Ae = A[v]



            int_A2 = area / 12.0 * (
                2.0 * (Ae[0] ** 2 + Ae[1] ** 2 + Ae[2] ** 2)
                + 2.0 * (Ae[0] * Ae[1] + Ae[1] * Ae[2] + Ae[2] * Ae[0])
            )
            P += coeff * int_A2

        return P

    def solve_nonlinear(
        self,
        material_map: dict,
        source_func,
        bc_nodes: dict,
        max_iter: int = 20,
        tol: float = 1.0e-6,
    ) -> np.ndarray:

        def nu_linear(x, y):
            tag = self._get_element_tag_at_point(x, y)
            if tag in material_map:
                return 1.0 / (material_map[tag].MU0 * material_map[tag].mu_r_init)
            return 1.0 / self.MU0

        K, F = self.assemble_linear(nu_linear, source_func)
        K, F = self.apply_dirichlet(K, F, bc_nodes)
        A = self.solve_linear(K, F)

        for it in range(max_iter):

            Bx, By = self.compute_b_field_at_nodes(A)
            B_mag = np.sqrt(Bx * Bx + By * By)

            def nu_nonlinear(x, y):
                tag = self._get_element_tag_at_point(x, y)
                if tag in material_map:
                    return material_map[tag].reluctivity(B_mag)
                return 1.0 / self.MU0

            K, F = self.assemble_linear(nu_nonlinear, source_func)
            K, F = self.apply_dirichlet(K, F, bc_nodes)
            A_new = self.solve_linear(K, F)

            delta = np.linalg.norm(A_new - A) / (np.linalg.norm(A_new) + 1.0e-12)
            A = A_new
            if delta < tol:
                break

        return A

    def _get_element_tag_at_point(self, x: float, y: float) -> int:

        dists = np.sum((self.mesh.nodes - np.array([x, y])) ** 2, axis=1)
        nearest = np.argmin(dists)

        return self.mesh.node_tags[nearest] if nearest < len(self.mesh.node_tags) else 0
