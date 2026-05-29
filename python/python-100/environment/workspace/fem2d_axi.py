"""
fem2d_axi.py
================================================================================
2D 轴对称电磁场有限元求解器：电机截面磁矢势分析

核心科学内容:
  1. 轴对称静磁问题的控制方程（二维Poisson-like方程）:

        ∂/∂x ( ν ∂A_z/∂x ) + ∂/∂y ( ν ∂A_z/∂y ) = -J_z

     其中 A_z 为轴向磁矢势分量，J_z 为轴向电流密度 [A/m^2]，
     ν = 1/μ 为磁阻率 [m/H].

     对于非线性材料，ν = ν(|∇A_z|) 为磁通密度的函数:
        B_x =  ∂A_z/∂y,   B_y = -∂A_z/∂x
        |B| = sqrt( (∂A_z/∂x)^2 + (∂A_z/∂y)^2 )
        ν = H(|B|) / |B| = 1/μ_apparent

  2. 三角形 P1 Lagrange 有限元离散:

        A_z(x,y) ≈ Σ_j A_j N_j(x,y)

     其中 N_j 为节点 j 的线性帽函数（hat function），在包含节点 j 的
     所有三角形上分段线性，在节点 j 处为 1，在其他节点为 0.

  3. Galerkin 弱形式:

        Σ_j A_j ∫_Ω_e ν ∇N_i · ∇N_j dΩ = ∫_Ω_e J_z N_i dΩ

     单元刚度矩阵（3×3）:
        K_{ij}^{(e)} = ∫_{T_e} ν ∇N_i · ∇N_j dA

     单元载荷向量（3×1）:
        F_i^{(e)} = ∫_{T_e} J_z N_i dA

     对于 P1 元，∇N_i 在单元内为常数:
        ∇N_i = (1/(2|Δ|)) [ (y_j - y_k), (x_k - x_j) ]^T

     其中 (i,j,k) 为三角形顶点的轮换下标。

  4. 非线性 Newton-Raphson 迭代:

        残差:   R(A) = K(A) A - F
        雅可比: J(A) = K(A) + ∂K/∂A * A

     迭代公式:  A^{(k+1)} = A^{(k)} - J^{-1} R(A^{(k)})

  5. Maxwell 应力张量与电磁转矩:

        T = (1/μ) [ B⊗B - 0.5 |B|^2 I ]

     电磁力密度:  f = ∇ · T
     电磁转矩（二维）:  τ_z = ∮_Γ r × (T · n) dl
                    = ∮_Γ (x T_{yx} - y T_{xx}) dl

     简化公式（气隙中径向/切向分量）:
        τ = (L/μ_0) ∮ r^2 B_r B_θ dl

  6. 涡流损耗计算（时谐场近似）:

        P_eddy = (1/(2σ)) ∫ |J_e|^2 dV
        J_e = -j ω σ A_z  （时谐场，复数形式）
        P_eddy = (ω^2 σ / 2) ∫ |A_z|^2 dV
================================================================================
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class FEM2DAxi:
    """
    2D 轴对称电磁场有限元求解器（三角形 P1 元）。
    """

    MU0 = 4.0 * np.pi * 1.0e-7

    def __init__(self, mesh):
        """
        参数:
            mesh : Mesh2D 对象
        """
        from mesh_engine import Mesh2D

        if not isinstance(mesh, Mesh2D):
            raise TypeError("mesh 必须是 Mesh2D 实例")
        self.mesh = mesh
        self.n_dof = mesh.n_nodes()

    def _shape_gradients(self, elem_idx: int) -> tuple:
        """
        计算单元 elem_idx 中三个线性基函数的梯度（常数向量）。

        返回:
            grads : (3, 2) 数组, grads[i] = [∂N_i/∂x, ∂N_i/∂y]
            area  : 单元面积
        """
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

        # 基函数梯度
        grads = np.zeros((3, 2))
        grads[0, 0] = (y2 - y3) / (2.0 * area)
        grads[0, 1] = (x3 - x2) / (2.0 * area)
        grads[1, 0] = (y3 - y1) / (2.0 * area)
        grads[1, 1] = (x1 - x3) / (2.0 * area)
        grads[2, 0] = (y1 - y2) / (2.0 * area)
        grads[2, 1] = (x2 - x1) / (2.0 * area)

        return grads, area

    def assemble_linear(self, nu_func, source_func, elem_tags_filter: set = None) -> tuple:
        """
        组装线性系统的稀疏刚度矩阵和载荷向量。

        参数:
            nu_func          : callable(x, y) -> float, 磁阻率
            source_func      : callable(x, y) -> float, 源电流密度 J_z
            elem_tags_filter : set, 仅组装指定标签的单元（None表示全部）

        返回:
            K : scipy.sparse.csr_matrix (n_dof, n_dof)
            F : np.ndarray (n_dof,)
        """
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

            # 单元质心（用于计算材料参数）
            centroid = np.mean(self.mesh.nodes[v], axis=0)
            nu_val = nu_func(centroid[0], centroid[1])

            # TODO: 请实现单元刚度矩阵和载荷向量的计算与全局组装
            raise NotImplementedError("Hole_2: 需实现2D FEM刚度矩阵组装与载荷向量计算")

        K = csr_matrix((data, (row_idx, col_idx)), shape=(n_dof, n_dof))
        return K, F

    def apply_dirichlet(self, K: csr_matrix, F: np.ndarray, bc_nodes: dict) -> tuple:
        """
        应用 Dirichlet 边界条件（大数罚函数法）.

        参数:
            bc_nodes : dict {node_idx: value}
        """
        K = K.tolil()
        for idx, val in bc_nodes.items():
            if idx < 0 or idx >= self.n_dof:
                raise ValueError(f"Dirichlet节点索引越界: {idx}")
            penalty = 1.0e16
            K[idx, idx] = penalty
            F[idx] = penalty * val
        return K.tocsr(), F

    def solve_linear(self, K: csr_matrix, F: np.ndarray) -> np.ndarray:
        """求解线性系统 K A = F."""
        # 数值稳定性检查
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
        """
        计算每个节点上的磁通密度 B = [B_x, B_y].

        采用单元常数梯度向节点投影（面积加权平均）:
            B_x = ∂A_z/∂y,   B_y = -∂A_z/∂x
        """
        n_dof = self.n_dof
        Bx = np.zeros(n_dof)
        By = np.zeros(n_dof)
        weight = np.zeros(n_dof)

        for e in range(self.mesh.n_elements()):
            v = self.mesh.elements[e]
            grads, area = self._shape_gradients(e)

            # 单元内 ∇A = Σ A_j ∇N_j
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

        # 归一化
        safe_w = np.where(weight < 1.0e-14, 1.0, weight)
        Bx /= safe_w
        By /= safe_w
        return Bx, By

    def compute_maxwell_stress_tensor(self, Bx: np.ndarray, By: np.ndarray) -> tuple:
        """
        计算 Maxwell 应力张量的节点值.

        T = (1/μ) [ B_x^2 - 0.5|B|^2     B_x B_y         ]
                   [ B_x B_y             B_y^2 - 0.5|B|^2 ]

        返回:
            Txx, Txy, Tyy : 各分量的节点值数组
        """
        B2 = Bx * Bx + By * By
        factor = 1.0 / self.MU0
        Txx = factor * (Bx * Bx - 0.5 * B2)
        Txy = factor * (Bx * By)
        Tyy = factor * (By * By - 0.5 * B2)
        return Txx, Txy, Tyy

    def compute_electromagnetic_torque(
        self, A: np.ndarray, radius_airgap_inner: float, radius_airgap_outer: float
    ) -> float:
        """
        基于 Maxwell 应力张量计算气隙中的电磁转矩（2D简化）.

        在气隙中间圆 r = (R_ro + R_si)/2 上积分:

            τ = L * ∮ r × f_surf dl
              = L * ∮ r^2 * (T_{rθ}) dθ

        其中 T_{rθ} = (1/μ_0) B_r B_θ 为切向应力分量。

        简化计算（基于节点值）:
            τ ≈ L * Σ_i (r_i^2 * T_{rθ,i} * Δθ_i)
        """
        Bx, By = self.compute_b_field_at_nodes(A)
        Txx, Txy, Tyy = self.compute_maxwell_stress_tensor(Bx, By)

        # 选取气隙中间附近的节点
        r_mid = 0.5 * (radius_airgap_inner + radius_airgap_outer)
        tol = 0.5 * (radius_airgap_outer - radius_airgap_inner)

        torque = 0.0
        count = 0
        L = 1.0  # 单位轴向长度

        for i in range(self.n_dof):
            x, y = self.mesh.nodes[i]
            r = np.sqrt(x * x + y * y)
            if abs(r - r_mid) < tol and r > 1.0e-10:
                theta = np.arctan2(y, x)
                # 局部坐标变换到极坐标
                # B_r = Bx cosθ + By sinθ
                # B_θ = -Bx sinθ + By cosθ
                Br = Bx[i] * np.cos(theta) + By[i] * np.sin(theta)
                Bt = -Bx[i] * np.sin(theta) + By[i] * np.cos(theta)
                Trt = (1.0 / self.MU0) * Br * Bt
                torque += r * r * Trt
                count += 1

        if count > 0:
            # 平均后乘以圆周
            torque = torque / count * 2.0 * np.pi * L
        return float(torque)

    def compute_magnetic_energy(self, A: np.ndarray, nu_func) -> float:
        """
        计算磁场储能:

            W_m = 0.5 * ∫ ν |B|^2 dA
                = 0.5 * A^T K A
        """
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
        """
        计算涡流损耗（时谐场近似）:

            P_eddy = (ω^2 σ / 2) ∫ |A_z|^2 dA

        对于三角形 P1 元:
            ∫ N_i N_j dA = |Δ| / 12  (i ≠ j)
            ∫ N_i^2 dA = |Δ| / 6
        """
        P = 0.0
        coeff = 0.5 * omega * omega * sigma
        for e in range(self.mesh.n_elements()):
            if elem_tags_filter is not None and self.mesh.elem_tags[e] not in elem_tags_filter:
                continue

            v = self.mesh.elements[e]
            _, area = self._shape_gradients(e)
            Ae = A[v]
            # ∫ A^2 dA ≈ area * (A1^2 + A2^2 + A3^2 + A1*A2 + A2*A3 + A3*A1) / 12
            # 更简洁：A_z 在单元内线性变化，|A|^2 的积分可用节点值表示
            # 质量矩阵近似
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
        """
        非线性 Newton-Raphson 求解器。

        参数:
            material_map : dict {tag: NonlinearMagneticMaterial}
            source_func  : callable(x,y) -> float
            bc_nodes     : Dirichlet边界条件
        """
        # 初始猜测（线性材料求解）
        def nu_linear(x, y):
            tag = self._get_element_tag_at_point(x, y)
            if tag in material_map:
                return 1.0 / (material_map[tag].MU0 * material_map[tag].mu_r_init)
            return 1.0 / self.MU0

        K, F = self.assemble_linear(nu_linear, source_func)
        K, F = self.apply_dirichlet(K, F, bc_nodes)
        A = self.solve_linear(K, F)

        for it in range(max_iter):
            # 基于当前解更新磁阻率
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
        """粗略估计点 (x,y) 所在单元的标签（基于最近节点）."""
        # 简单方法：找最近节点并返回其标签
        dists = np.sum((self.mesh.nodes - np.array([x, y])) ** 2, axis=1)
        nearest = np.argmin(dists)
        # 使用最近节点的标签（近似）
        return self.mesh.node_tags[nearest] if nearest < len(self.mesh.node_tags) else 0
