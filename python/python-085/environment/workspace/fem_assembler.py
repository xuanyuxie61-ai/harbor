"""
fem_assembler.py
有限元刚度矩阵与残差组装模块
融合种子项目：
  - 871_plasma_matrix（非线性 Jacobian 矩阵组装模式）
  - 1379_usa_matrix（稀疏矩阵构造与拓扑模式）
"""
import numpy as np
from typing import Tuple, Optional
from mesh_generator import TriMesh2D


class ElasticFEM2D:
    r"""
    二维线弹性有限元求解器（平面应变）。

    本构关系（Hooke 定律，平面应变）：
    \sigma = D : \varepsilon

    D = \frac{E(1-\nu)}{(1+\nu)(1-2\nu)}
    \begin{bmatrix}
    1 & \frac{\nu}{1-\nu} & 0 \\
    \frac{\nu}{1-\nu} & 1 & 0 \\
    0 & 0 & \frac{1-2\nu}{2(1-\nu)}
    \end{bmatrix}

    其中 E 为杨氏模量，\nu 为泊松比。
    """

    def __init__(self, mesh: TriMesh2D, young: float = 1e11, nu: float = 0.3,
                 thickness: float = 1.0):
        self.mesh = mesh
        self.young = young
        self.nu = nu
        self.thickness = thickness
        self._D = self._compute_D_matrix()
        self._K: Optional[np.ndarray] = None

    def _compute_D_matrix(self) -> np.ndarray:
        r"""
        计算平面应变弹性矩阵 D（3x3）。
        """
        # [HOLE 1]: 请实现平面应变弹性矩阵 D 的计算公式。
        # 提示：D = factor * [[1, nu/(1-nu), 0], [nu/(1-nu), 1, 0], [0, 0, (1-2*nu)/(2*(1-nu))]]
        # 其中 factor = E * (1-nu) / ((1+nu) * (1-2*nu))
        pass

    def _element_stiffness(self, elem_id: int) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        计算单元刚度矩阵 k_e（6x6）。
        公式：k_e = t * A_e * B_e^T * D * B_e
        其中 B_e 是应变-位移矩阵（3x6）。
        返回 (k_e, B_e)。
        """
        dN = self.mesh.shape_functions_gradients(elem_id)
        # B 矩阵：3 x 6
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
        r"""
        组装全局刚度矩阵 K（2*n_nodes x 2*n_nodes）。
        采用稀疏稠密矩阵格式，融合 usa_matrix 的稀疏构造思想。
        """
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
        r"""
        从位移场 u 计算单元应变和应力。
        \varepsilon_e = B_e u_e
        \sigma_e = D \varepsilon_e
        返回 (strain_array, stress_array)，形状均为 (n_elements, 3)。
        """
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
        r"""
        计算残差：r = K u - f_ext
        融合 plasma_matrix 的残差构造思想。
        """
        K = self.assemble_global_stiffness()
        return K @ u - f_ext

    def compute_internal_force(self, u: np.ndarray) -> np.ndarray:
        r"""计算内力向量 f_int = K u。"""
        K = self.assemble_global_stiffness()
        return K @ u

    def compute_strain_energy(self, u: np.ndarray) -> float:
        r"""
        计算应变能：
        W = 0.5 * u^T K u
        """
        K = self.assemble_global_stiffness()
        return 0.5 * float(u @ (K @ u))

    def apply_dirichlet_bc(self, K: np.ndarray, F: np.ndarray,
                           bc_nodes: np.ndarray, bc_values: np.ndarray,
                           dof_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        施加 Dirichlet 边界条件（位移约束）。
        采用大数罚因子法，保证数值稳定性。
        dof_mask: (n_bc, 2) 的布尔数组，True 表示该自由度需要固定。
        """
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
    r"""
    计算接触节点在当前位移下的间隙（gap）。
    g_n(i) = (y_i + u_y(i)) - y_rigid
    若 g_n < 0，表示穿透。
    Signorini 条件要求 g_n \ge 0。
    """
    gaps = np.zeros(len(contact_nodes))
    for idx, node in enumerate(contact_nodes):
        y_curr = mesh.nodes[node, 1] + u[2 * node + 1]
        gaps[idx] = y_curr - rigid_surface_y
    return gaps


def assemble_contact_normals(mesh: TriMesh2D, contact_nodes: np.ndarray) -> np.ndarray:
    r"""
    计算接触边界外法向（简化版：对底部边界取向下法向）。
    对于水平刚性基础，法向为 n = [0, -1]^T。
    返回形状 (len(contact_nodes), 2)。
    """
    nvec = np.zeros((len(contact_nodes), 2))
    nvec[:, 1] = -1.0
    return nvec
