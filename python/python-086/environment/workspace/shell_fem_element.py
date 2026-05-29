# -*- coding: utf-8 -*-
"""
shell_fem_element.py
壳体有限元离散化与刚度矩阵核心

融合种子项目:
  - 417_fem3d_pack: 3D有限元基函数、坐标变换、行列式计算

科学背景:
  采用 Donnell-Mushtari-Vlasov (DMV) 壳理论对圆柱壳进行离散。
  中曲面位移场 u = [u, v, w]ᵀ，其中 u 为轴向，v 为环向，w 为法向。

  薄膜应变:
    ε_x  = ∂u/∂x
    ε_θ  = (1/R)(∂v/∂θ + w)
    γ_xθ = ∂v/∂x + (1/R)∂u/∂θ

  弯曲应变 (曲率变化):
    κ_x  = -∂²w/∂x²
    κ_θ  = -(1/R²)(∂²w/∂θ² - ∂v/∂θ)
    κ_xθ = -(1/R)(∂²w/∂x∂θ - (3/4)∂v/∂x + (1/4R)∂u/∂θ)

  本构关系 (线弹性各向同性):
    N = C_m · ε,   M = C_b · κ
    其中薄膜刚度矩阵 C_m = (Et/(1-ν²)) * [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
    弯曲刚度矩阵 C_b = (Et³/(12(1-ν²))) * [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]

  单元刚度矩阵 (局部坐标):
    k_e = ∫∫ (B_mᵀ C_m B_m + B_bᵀ C_b B_b) R dθ dx

  采用 3节点三角形单元，每个节点自由度 DOF = [u, v, w, ∂w/∂x, ∂w/∂θ]
  (简化为每个节点 3 DOF: u, v, w，通过离散差分近似曲率)
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple


class ShellMaterial:
    """
    壳体线弹性材料参数
    """

    def __init__(self, E: float, nu: float, rho: float = 7850.0):
        """
        Parameters
        ----------
        E : float
            杨氏模量 (Pa)
        nu : float
            泊松比
        rho : float
            密度 (kg/m³)
        """
        if E <= 0.0:
            raise ValueError("杨氏模量必须为正")
        if not (0.0 <= nu < 0.5):
            raise ValueError("泊松比必须在 [0, 0.5) 范围内")
        self.E = float(E)
        self.nu = float(nu)
        self.rho = float(rho)

    def extensional_rigidity(self, t: float) -> float:
        """
        薄膜刚度 C = Et / (1 - ν²)
        """
        return self.E * t / (1.0 - self.nu ** 2)

    def bending_rigidity(self, t: float) -> float:
        """
        弯曲刚度 D = Et³ / (12(1 - ν²))
        """
        return self.E * t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    def membrane_matrix(self, t: float) -> np.ndarray:
        """
        薄膜刚度矩阵 C_m (3×3)
        """
        C = self.extensional_rigidity(t)
        nu = self.nu
        return C * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)]
        ])

    def bending_matrix(self, t: float) -> np.ndarray:
        """
        弯曲刚度矩阵 C_b (3×3)
        """
        D = self.bending_rigidity(t)
        nu = self.nu
        return D * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)]
        ])


class ShellFEModel:
    """
    壳体有限元模型组装器
    """

    def __init__(self, mesh, material: ShellMaterial):
        self.mesh = mesh
        self.mat = material
        self.n_dof_per_node = 3
        self.n_nodes = mesh.n_nodes
        self.n_dof = self.n_nodes * self.n_dof_per_node
        # 预计算单元几何信息
        self._compute_element_geometry()

    def _compute_element_geometry(self):
        """
        预计算每个三角形单元的:
          - 面积 A_e
          - 形函数对局部坐标的导数 dN/dξ, dN/dη
          - 雅可比矩阵 J 及其逆
        """
        self.elem_areas = np.zeros(self.mesh.n_elem)
        self.elem_dNdx = np.zeros((self.mesh.n_elem, 3, 3))
        self.elem_dNdy = np.zeros((self.mesh.n_elem, 3, 3))
        for eid in range(self.mesh.n_elem):
            nodes = self.mesh.elements[eid]
            coords = self.mesh.nodes[nodes]  # (3, 3)
            # 局部坐标假设: 节点1=(0,0), 节点2=(1,0), 节点3=(0,1)
            # 雅可比 J = [∂x/∂ξ, ∂x/∂η; ∂y/∂ξ, ∂y/∂η]
            # 使用参数域坐标: x_param = θ, y_param = z (轴向)
            # 从三维反解参数
            R = self.mesh.geom.R
            theta = np.arctan2(coords[:, 1], coords[:, 0])
            x_axial = coords[:, 2]
            # 局部三角形坐标 (取前两个参数)
            x1, y1 = theta[0], x_axial[0]
            x2, y2 = theta[1], x_axial[1]
            x3, y3 = theta[2], x_axial[2]
            J = np.array([
                [x2 - x1, x3 - x1],
                [y2 - y1, y3 - y1]
            ])
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-14:
                detJ = 1e-14
            self.elem_areas[eid] = 0.5 * abs(detJ) * R  # 中曲面面积
            Jinv = np.linalg.inv(J)
            # 形函数导数 (参考单元)
            dN_dxi = np.array([-1.0, 1.0, 0.0])
            dN_deta = np.array([-1.0, 0.0, 1.0])
            dNdx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
            dNdy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
            self.elem_dNdx[eid, :, 0] = dNdx
            self.elem_dNdx[eid, :, 1] = dNdy
            self.elem_dNdx[eid, :, 2] = np.zeros(3)  # 法向w的导数占位

    def _b_matrix_membrane(self, eid: int) -> np.ndarray:
        """
        薄膜应变-位移矩阵 B_m (3×9) for 3-node三角形
        自由度排序: [u1,v1,w1, u2,v2,w2, u3,v3,w3]

        科学背景 (Donnell-Mushtari-Vlasov 壳理论):
          ε_x  = ∂u/∂x
          ε_θ  = (1/R)(∂v/∂θ + w)
          γ_xθ = ∂v/∂x + (1/R)∂u/∂θ

        提示:
          - dNdx = self.elem_dNdx[eid, :, 0] 表示 dN/dθ
          - dNdy = self.elem_dNdx[eid, :, 1] 表示 dN/dx
          - R = self.mesh.geom.R
        """
        # === HOLE 1 ===
        # 请实现薄膜应变-位移矩阵 B_m (3×9)
        # 该矩阵的正确性直接影响 assemble_linear_stiffness 和
        # assemble_geometric_stiffness 的结果，进而影响 nonlinear_solver
        # 和 arc_length_tracker 中的 Newton-Raphson 迭代收敛性。
        raise NotImplementedError("Hole 1: 请实现 _b_matrix_membrane")
        # ==============

    def _b_matrix_bending(self, eid: int) -> np.ndarray:
        """
        弯曲应变-位移矩阵 B_b (3×9)
        采用离散Kirchhoff假设，曲率由 w 的二阶导数近似。
        对于三角形单元，使用面积坐标下的常曲率近似。
        """
        R = self.mesh.geom.R
        A = self.elem_areas[eid]
        if A < 1e-20:
            A = 1e-20
        # 简化的常曲率矩阵 (基于节点w的离散Laplacian)
        # κ_x = -∂²w/∂x², κ_θ = -(1/R²)∂²w/∂θ²
        # 采用节点平均曲率插值
        Bb = np.zeros((3, 9))
        nodes = self.mesh.elements[eid]
        coords = self.mesh.nodes[nodes]
        # 计算三角形边向量
        e1 = coords[1] - coords[0]
        e2 = coords[2] - coords[1]
        e3 = coords[0] - coords[2]
        # 法向分量用于曲率近似
        for i in range(3):
            col = i * 3 + 2
            # 简化: 将曲率近似为与相邻边长度成反比
            le = np.linalg.norm([e1, e2, e3][i])
            if le > 0:
                Bb[0, col] = -1.0 / (le * R)  # κ_x
                Bb[1, col] = -1.0 / (le * R ** 2)  # κ_θ
        return Bb

    def assemble_linear_stiffness(self) -> csr_matrix:
        """
        组装线性刚度矩阵 K (n_dof × n_dof)

        K = Σ_e (B_mᵀ C_m B_m + B_bᵀ C_b B_b) * A_e

        Returns
        -------
        K : csr_matrix
        """
        t = self.mesh.geom.t
        Cm = self.mat.membrane_matrix(t)
        Cb = self.mat.bending_matrix(t)
        row_idx = []
        col_idx = []
        data = []
        for eid in range(self.mesh.n_elem):
            nodes = self.mesh.elements[eid]
            Bm = self._b_matrix_membrane(eid)
            Bb = self._b_matrix_bending(eid)
            Ae = self.elem_areas[eid]
            ke = (Bm.T @ Cm @ Bm + Bb.T @ Cb @ Bb) * Ae
            # 组装到全局
            dofs = []
            for nid in nodes:
                dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
            dofs = np.array(dofs, dtype=int)
            for i in range(9):
                for j in range(9):
                    if abs(ke[i, j]) > 1e-18:
                        row_idx.append(dofs[i])
                        col_idx.append(dofs[j])
                        data.append(ke[i, j])
        K = csr_matrix((data, (row_idx, col_idx)), shape=(self.n_dof, self.n_dof))
        return K

    def assemble_geometric_stiffness(self, disp: np.ndarray) -> csr_matrix:
        """
        组装几何刚度矩阵 K_g (应力刚度矩阵)

        基于当前位移场 disp 计算薄膜应力，进而构造 K_g。
        线性化假设下:
          K_g = ∫ Gᵀ σ G dA
        其中 G 为位移梯度矩阵，σ 为 Cauchy 应力张量。

        Parameters
        ----------
        disp : (n_dof,) ndarray
            当前位移向量

        Returns
        -------
        Kg : csr_matrix
        """
        t = self.mesh.geom.t
        Cm = self.mat.membrane_matrix(t)
        row_idx = []
        col_idx = []
        data = []
        for eid in range(self.mesh.n_elem):
            nodes = self.mesh.elements[eid]
            Bm = self._b_matrix_membrane(eid)
            Ae = self.elem_areas[eid]
            dofs = []
            for nid in nodes:
                dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
            dofs = np.array(dofs, dtype=int)
            ue = disp[dofs]
            # 薄膜应变
            eps = Bm @ ue
            # 应力 (平面应力)
            sigma = Cm @ eps
            Nx, Ntheta, Nxtheta = sigma[0], sigma[1], sigma[2]
            # 简化的几何刚度 (仅法向w的自由度耦合)
            # 基于 von Karman 大挠度理论
            R = self.mesh.geom.R
            for i in range(3):
                for j in range(3):
                    ii = nodes[i] * 3 + 2
                    jj = nodes[j] * 3 + 2
                    val = (Nx * 1.0 + Ntheta / (R ** 2) + Nxtheta * 0.5) * Ae / 9.0
                    if abs(val) > 1e-18:
                        row_idx.append(ii)
                        col_idx.append(jj)
                        data.append(val)
        Kg = csr_matrix((data, (row_idx, col_idx)), shape=(self.n_dof, self.n_dof))
        return Kg

    def internal_force(self, disp: np.ndarray) -> np.ndarray:
        """
        计算内力向量 F_int = K(u) · u

        Parameters
        ----------
        disp : (n_dof,) ndarray

        Returns
        -------
        fint : (n_dof,) ndarray
        """
        t = self.mesh.geom.t
        Cm = self.mat.membrane_matrix(t)
        Cb = self.mat.bending_matrix(t)
        fint = np.zeros(self.n_dof)
        for eid in range(self.mesh.n_elem):
            nodes = self.mesh.elements[eid]
            Bm = self._b_matrix_membrane(eid)
            Bb = self._b_matrix_bending(eid)
            Ae = self.elem_areas[eid]
            dofs = []
            for nid in nodes:
                dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
            dofs = np.array(dofs, dtype=int)
            ue = disp[dofs]
            eps = Bm @ ue
            kap = Bb @ ue
            N = Cm @ eps
            M = Cb @ kap
            fe = (Bm.T @ N + Bb.T @ M) * Ae
            fint[dofs] += fe
        return fint
