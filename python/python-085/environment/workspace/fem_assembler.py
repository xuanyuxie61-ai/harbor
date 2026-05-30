import numpy as np
from typing import Tuple, Optional
from mesh_generator import TriMesh2D


class ElasticFEM2D:

    def __init__(self, mesh: TriMesh2D, young: float = 1e11, nu: float = 0.3,
                 thickness: float = 1.0):
        self.mesh = mesh
        self.young = young
        self.nu = nu
        self.thickness = thickness
        self._D = self._compute_D_matrix()
        self._K: Optional[np.ndarray] = None

    def _compute_D_matrix(self) -> np.ndarray:



        pass

    def _element_stiffness(self, elem_id: int) -> Tuple[np.ndarray, np.ndarray]:
        dN = self.mesh.shape_functions_gradients(elem_id)

        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2 * i] = dN[i, 0]
            B[1, 2 * i + 1] = dN[i, 1]
            B[2, 2 * i] = dN[i, 1]
            B[2, 2 * i + 1] = dN[i, 0]
        area = abs(self.mesh.compute_areas()[elem_id])
        k_e = self.thickness * area * (B.T @ self._D @ B)
        return k_e, B

    def assemble_global_stiffness(self, penalty_scale: float = 1.0) -> np.ndarray:
        if self._K is not None:
            return self._K
        n_dof = 2 * self.mesh.n_nodes
        K = np.zeros((n_dof, n_dof))
        for elem_id in range(self.mesh.n_elements):
            k_e, _ = self._element_stiffness(elem_id)
            tri = self.mesh.elements[elem_id]
            dof_map = np.zeros(6, dtype=int)
            for i in range(3):
                dof_map[2 * i] = 2 * tri[i]
                dof_map[2 * i + 1] = 2 * tri[i] + 1
            for i in range(6):
                for j in range(6):
                    K[dof_map[i], dof_map[j]] += k_e[i, j]
        self._K = K * penalty_scale
        return self._K

    def compute_strain_stress(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if u.shape[0] != 2 * self.mesh.n_nodes:
            raise ValueError("Displacement vector size mismatch")
        strain = np.zeros((self.mesh.n_elements, 3))
        stress = np.zeros((self.mesh.n_elements, 3))
        for elem_id in range(self.mesh.n_elements):
            _, B = self._element_stiffness(elem_id)
            tri = self.mesh.elements[elem_id]
            ue = np.zeros(6)
            for i in range(3):
                ue[2 * i] = u[2 * tri[i]]
                ue[2 * i + 1] = u[2 * tri[i] + 1]
            eps = B @ ue
            strain[elem_id] = eps
            stress[elem_id] = self._D @ eps
        return strain, stress

    def compute_residual(self, u: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        K = self.assemble_global_stiffness()
        return K @ u - f_ext

    def compute_internal_force(self, u: np.ndarray) -> np.ndarray:
        K = self.assemble_global_stiffness()
        return K @ u

    def compute_strain_energy(self, u: np.ndarray) -> float:
        K = self.assemble_global_stiffness()
        return 0.5 * float(u @ (K @ u))

    def apply_dirichlet_bc(self, K: np.ndarray, F: np.ndarray,
                           bc_nodes: np.ndarray, bc_values: np.ndarray,
                           dof_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        Kc = K.copy()
        Fc = F.copy()
        penalty = 1e16 * np.max(np.abs(K.diagonal()))
        if penalty == 0:
            penalty = 1e16
        if dof_mask is None:
            dof_mask = np.ones((len(bc_nodes), 2), dtype=bool)
        for idx, node in enumerate(bc_nodes):
            for dof in range(2):
                if dof_mask[idx, dof]:
                    gdof = 2 * node + dof
                    Kc[gdof, gdof] += penalty
                    Fc[gdof] += penalty * bc_values[idx, dof]
        return Kc, Fc


def assemble_contact_gaps(mesh: TriMesh2D, u: np.ndarray,
                          contact_nodes: np.ndarray,
                          rigid_surface_y: float = 0.0) -> np.ndarray:
    gaps = np.zeros(len(contact_nodes))
    for idx, node in enumerate(contact_nodes):
        y_curr = mesh.nodes[node, 1] + u[2 * node + 1]
        gaps[idx] = y_curr - rigid_surface_y
    return gaps


def assemble_contact_normals(mesh: TriMesh2D, contact_nodes: np.ndarray) -> np.ndarray:
    nvec = np.zeros((len(contact_nodes), 2))
    nvec[:, 1] = -1.0
    return nvec
