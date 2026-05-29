"""
fem_core.py
二维线弹性有限元核心求解模块

融合来源：
- 408_fem2d_poisson_rectangle: 有限元框架（T6二次三角形、刚度矩阵组装、边界条件）
- 970_r8blt: 带状下三角矩阵求解器

科学背景：
骨骼在力学载荷下的变形可用线弹性理论描述。

控制方程（强形式）：
    -∇·σ = f   in Ω
    σ = C : ε
    ε = 0.5 * (∇u + (∇u)^T)
    u = g    on Γ_D
    σ·n = t  on Γ_N

弱形式（Galerkin）：
    ∫_Ω σ(u) : ε(v) dΩ = ∫_Ω f·v dΩ + ∫_{Γ_N} t·v dS

其中 C 为弹性张量，对于各向同性材料：
    C_{ijkl} = λ δ_{ij} δ_{kl} + μ (δ_{ik} δ_{jl} + δ_{il} δ_{jk})

平面应力近似下的弹性矩阵 D（3x3）：
    D = E/(1-ν²) * [[1, ν, 0],
                    [ν, 1, 0],
                    [0, 0, (1-ν)/2]]

T6 二次三角形单元（6节点）：
    3个角节点 + 3个中边节点
    形函数在参考单元上为二次多项式
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional, Callable


# ===================================================================
# T6 二次三角形形函数（来自 408_fem2d_poisson_rectangle 的 qbf）
# ===================================================================
def t6_basis_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在参考三角形 {(0,0),(1,0),(0,1)} 上求 T6 二次形函数及其导数。

    节点顺序（参考单元）：
        N1 = (0, 0)    - 角节点
        N2 = (1, 0)    - 角节点
        N3 = (0, 1)    - 角节点
        N4 = (0.5, 0)  - 中边节点
        N5 = (0.5, 0.5) - 中边节点
        N6 = (0, 0.5)  - 中边节点

    形函数：
        φ1 = (1 - ξ - η)(1 - 2ξ - 2η)
        φ2 = ξ(2ξ - 1)
        φ3 = η(2η - 1)
        φ4 = 4ξ(1 - ξ - η)
        φ5 = 4ξη
        φ6 = 4η(1 - ξ - η)

    Parameters
    ----------
    xi, eta : float
        参考坐标

    Returns
    -------
    phi : np.ndarray, shape (6,)
        形函数值
    dphi_dxi : np.ndarray, shape (6,)
        ξ方向导数
    dphi_deta : np.ndarray, shape (6,)
        η方向导数
    """
    phi = np.zeros(6)
    dphi_dxi = np.zeros(6)
    dphi_deta = np.zeros(6)

    phi[0] = (1.0 - xi - eta) * (1.0 - 2.0 * xi - 2.0 * eta)
    phi[1] = xi * (2.0 * xi - 1.0)
    phi[2] = eta * (2.0 * eta - 1.0)
    phi[3] = 4.0 * xi * (1.0 - xi - eta)
    phi[4] = 4.0 * xi * eta
    phi[5] = 4.0 * eta * (1.0 - xi - eta)

    dphi_dxi[0] = -3.0 + 4.0 * xi + 4.0 * eta
    dphi_dxi[1] = 4.0 * xi - 1.0
    dphi_dxi[2] = 0.0
    dphi_dxi[3] = 4.0 - 8.0 * xi - 4.0 * eta
    dphi_dxi[4] = 4.0 * eta
    dphi_dxi[5] = -4.0 * eta

    dphi_deta[0] = -3.0 + 4.0 * xi + 4.0 * eta
    dphi_deta[1] = 0.0
    dphi_deta[2] = 4.0 * eta - 1.0
    dphi_deta[3] = -4.0 * xi
    dphi_deta[4] = 4.0 * xi
    dphi_deta[5] = 4.0 - 4.0 * xi - 8.0 * eta

    return phi, dphi_dxi, dphi_deta


def t6_physical_derivatives(xi: float, eta: float,
                            x_phys: np.ndarray, y_phys: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算物理坐标下的形函数值及其偏导数。

    使用等参映射的 Jacobian 矩阵：
        J = [[dx/dξ, dx/dη],
             [dy/dξ, dy/dη]]

    逆 Jacobian：
        J^{-1} = (1/detJ) * [[ dy/dη, -dx/dη],
                             [-dy/dξ,  dx/dξ]]

    物理导数：
        [dφ/dx] = J^{-1} [dφ/dξ]
        [dφ/dy]          [dφ/dη]
    """
    phi, dphi_dxi, dphi_deta = t6_basis_functions(xi, eta)

    # Jacobian
    dx_dxi = np.dot(dphi_dxi, x_phys)
    dx_deta = np.dot(dphi_deta, x_phys)
    dy_dxi = np.dot(dphi_dxi, y_phys)
    dy_deta = np.dot(dphi_deta, y_phys)

    detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
    if abs(detJ) < 1e-14:
        raise ValueError(f"Singular Jacobian in T6 element, detJ = {detJ}")

    # 逆 Jacobian 作用于形函数导数
    dphi_dx = (dy_deta * dphi_dxi - dy_dxi * dphi_deta) / detJ
    dphi_dy = (-dx_deta * dphi_dxi + dx_dxi * dphi_deta) / detJ

    return phi, dphi_dx, dphi_dy


# ===================================================================
# 弹性矩阵 D
# ===================================================================
def elastic_matrix_plane_stress(E: float, nu: float) -> np.ndarray:
    """
    平面应力问题的弹性矩阵 D（3x3）。

        D = E/(1-ν²) * [[1, ν, 0],
                        [ν, 1, 0],
                        [0, 0, (1-ν)/2]]
    """
    if E <= 0:
        raise ValueError("Young's modulus E must be positive.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("Poisson's ratio nu must be in (-1, 0.5).")

    factor = E / (1.0 - nu * nu)
    D = factor * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])
    return D


# ===================================================================
# 带状下三角矩阵求解器（来自 970_r8blt 的 r8blt_sl）
# ===================================================================
def r8blt_sl(n: int, ml: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    求解 A * x = b，其中 A 为带状下三角矩阵（R8BLT 存储格式）。

    存储格式：a[0, :] 为对角线，a[1, :] 为第1次对角线，...，a[ml, :] 为第ml次对角线。

    前向代入法：
        x_j = b_j / A_{jj}
        for i = j+1 ... min(j+ml, n):
            b_i = b_i - A_{ij} * x_j

    Parameters
    ----------
    n : int
        矩阵阶数
    ml : int
        下带宽
    a : np.ndarray, shape (ml+1, n)
        R8BLT 格式矩阵
    b : np.ndarray, shape (n,)
        右端向量

    Returns
    -------
    x : np.ndarray, shape (n,)
        解向量
    """
    if a.shape != (ml + 1, n):
        raise ValueError(f"Matrix shape mismatch: expected ({ml+1}, {n}), got {a.shape}")
    if b.shape != (n,):
        raise ValueError(f"RHS shape mismatch: expected ({n},), got {b.shape}")

    x = b.copy()
    for j in range(n):
        if abs(a[0, j]) < 1e-15:
            raise ValueError(f"Zero diagonal element at row {j}")
        x[j] = x[j] / a[0, j]
        ihi = min(j + ml + 1, n)
        for i in range(j + 1, ihi):
            x[i] = x[i] - a[i - j, j] * x[j]
    return x


def r8blt_mv(n: int, ml: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    计算带状下三角矩阵与向量的乘积 y = A * x。
    """
    y = np.zeros(n)
    for i in range(n):
        jlo = max(0, i - ml)
        for j in range(jlo, i + 1):
            y[i] += a[i - j, j] * x[j]
    return y


# ===================================================================
# 有限元求解器
# ===================================================================
class ElasticFEM2D:
    """
    二维线弹性有限元求解器（T6二次三角形单元）。
    """

    def __init__(self, node_xy: np.ndarray, element_node: np.ndarray,
                 element_area: np.ndarray, E_field: np.ndarray,
                 nu: float = 0.3, thickness: float = 1.0):
        """
        Parameters
        ----------
        node_xy : np.ndarray, shape (2, n_nodes)
            节点坐标
        element_node : np.ndarray, shape (6, n_elements)
            单元连接关系（0-based）
        element_area : np.ndarray, shape (n_elements,)
            单元面积
        E_field : np.ndarray, shape (n_nodes,)
            每个节点的弹性模量（非均匀材料）
        nu : float
            泊松比
        thickness : float
            截面厚度 (mm)
        """
        self.node_xy = node_xy
        self.element_node = element_node
        self.element_area = element_area
        self.E_field = E_field
        self.nu = nu
        self.thickness = thickness
        self.n_nodes = node_xy.shape[1]
        self.n_elements = element_node.shape[1]
        self.dof_per_node = 2
        self.n_dofs = self.n_nodes * self.dof_per_node

        # 求积规则（3点中边，精度3）
        self.quad_xi = np.array([0.5, 0.5, 0.0])
        self.quad_eta = np.array([0.0, 0.5, 0.5])
        self.quad_w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        self.nq = 3

    def _assemble_stiffness(self) -> csr_matrix:
        """
        组装全局刚度矩阵 K。

        对每个单元 e：
            K_e = ∫_{Ω_e} B^T D B dΩ
        其中 B 为应变-位移矩阵。
        """
        n_dofs = self.n_dofs
        n_elements = self.n_elements
        nq = self.nq

        row_indices = []
        col_indices = []
        data = []

        for e in range(n_elements):
            # 单元节点坐标
            enodes = self.element_node[:, e]
            x_phys = self.node_xy[0, enodes]
            y_phys = self.node_xy[1, enodes]

            # 单元平均弹性模量
            E_avg = np.mean(self.E_field[enodes])
            D = elastic_matrix_plane_stress(E_avg, self.nu)

            Ke = np.zeros((12, 12))  # 6 nodes * 2 dofs

            for q in range(nq):
                xi = self.quad_xi[q]
                eta = self.quad_eta[q]
                w = self.quad_w[q]

                phi, dphi_dx, dphi_dy = t6_physical_derivatives(
                    xi, eta, x_phys, y_phys)

                # Jacobian 行列式（用于面积元变换）
                _, dphi_dxi, dphi_deta = t6_basis_functions(xi, eta)
                dx_dxi = np.dot(dphi_dxi, x_phys)
                dx_deta = np.dot(dphi_deta, x_phys)
                dy_dxi = np.dot(dphi_dxi, y_phys)
                dy_deta = np.dot(dphi_deta, y_phys)
                detJ = dx_dxi * dy_deta - dx_deta * dy_dxi

                # 应变-位移矩阵 B (3 x 12)
                B = np.zeros((3, 12))
                for i in range(6):
                    B[0, 2 * i] = dphi_dx[i]      # ε_xx
                    B[1, 2 * i + 1] = dphi_dy[i]  # ε_yy
                    B[2, 2 * i] = dphi_dy[i]      # γ_xy
                    B[2, 2 * i + 1] = dphi_dx[i]

                # 积分权重 = w * |detJ| * thickness
                weight = w * abs(detJ) * self.thickness
                Ke += weight * (B.T @ D @ B)

            # 组装到全局矩阵
            for i in range(6):
                for j in range(6):
                    gi = enodes[i]
                    gj = enodes[j]
                    # x-x 耦合
                    row_indices.append(2 * gi)
                    col_indices.append(2 * gj)
                    data.append(Ke[2 * i, 2 * j])
                    # x-y 耦合
                    row_indices.append(2 * gi)
                    col_indices.append(2 * gj + 1)
                    data.append(Ke[2 * i, 2 * j + 1])
                    # y-x 耦合
                    row_indices.append(2 * gi + 1)
                    col_indices.append(2 * gj)
                    data.append(Ke[2 * i + 1, 2 * j])
                    # y-y 耦合
                    row_indices.append(2 * gi + 1)
                    col_indices.append(2 * gj + 1)
                    data.append(Ke[2 * i + 1, 2 * j + 1])

        K = csr_matrix((data, (row_indices, col_indices)), shape=(n_dofs, n_dofs))
        return K

    def _assemble_load(self, body_force: Optional[Callable] = None) -> np.ndarray:
        """
        组装载荷向量 F。

        默认：在顶部边界施加均匀分布的压缩载荷（模拟体重载荷）。
        """
        F = np.zeros(self.n_dofs)

        # 体力载荷（如重力）
        if body_force is not None:
            for e in range(self.n_elements):
                enodes = self.element_node[:, e]
                x_phys = self.node_xy[0, enodes]
                y_phys = self.node_xy[1, enodes]

                for q in range(self.nq):
                    xi = self.quad_xi[q]
                    eta = self.quad_eta[q]
                    w = self.quad_w[q]

                    phi, _, _ = t6_physical_derivatives(xi, eta, x_phys, y_phys)

                    # 物理坐标
                    xq = np.dot(phi, x_phys)
                    yq = np.dot(phi, y_phys)
                    fx, fy = body_force(xq, yq)

                    _, dphi_dxi, dphi_deta = t6_basis_functions(xi, eta)
                    dx_dxi = np.dot(dphi_dxi, x_phys)
                    dx_deta = np.dot(dphi_deta, x_phys)
                    dy_dxi = np.dot(dphi_dxi, y_phys)
                    dy_deta = np.dot(dphi_deta, y_phys)
                    detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
                    weight = w * abs(detJ) * self.thickness

                    for i in range(6):
                        F[2 * enodes[i]] += weight * fx * phi[i]
                        F[2 * enodes[i] + 1] += weight * fy * phi[i]

        return F

    def apply_dirichlet_bc(self, K: csr_matrix, F: np.ndarray,
                           bc_nodes: np.ndarray,
                           bc_values: Optional[np.ndarray] = None) -> Tuple[csr_matrix, np.ndarray]:
        """
        施加 Dirichlet 边界条件。

        直接消去法：将边界自由度对应的行列置为 1，右端项设为给定值。
        """
        K = K.tolil()
        F = F.copy()

        if bc_values is None:
            bc_values = np.zeros(len(bc_nodes) * 2)

        dof_list = []
        for node in bc_nodes:
            dof_list.append(2 * node)      # x 方向
            dof_list.append(2 * node + 1)  # y 方向

        for dof in dof_list:
            K[dof, :] = 0.0
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
            F[dof] = 0.0

        # 应用给定位移
        for i, node in enumerate(bc_nodes):
            F[2 * node] = bc_values[2 * i] if 2 * i < len(bc_values) else 0.0
            F[2 * node + 1] = bc_values[2 * i + 1] if 2 * i + 1 < len(bc_values) else 0.0

        return K.tocsr(), F

    def apply_neumann_bc(self, F: np.ndarray, boundary_elements: np.ndarray,
                         traction: np.ndarray) -> np.ndarray:
        """
        施加 Neumann 边界条件（面力）。

        Parameters
        ----------
        boundary_elements : np.ndarray
            边界单元索引
        traction : np.ndarray, shape (2,)
            面力向量 [tx, ty]
        """
        F = F.copy()
        # 简化为在边界节点上直接施加等效节点力
        # 实际实现需要对边界边进行线积分
        # 这里采用简化处理：将面力均匀分配到边界单元节点
        for e in boundary_elements:
            enodes = self.element_node[:, e]
            edge_len = 0.0
            # 用单元周长估算
            for i in range(6):
                n1 = enodes[i]
                n2 = enodes[(i + 1) % 6]
                dx = self.node_xy[0, n1] - self.node_xy[0, n2]
                dy = self.node_xy[1, n1] - self.node_xy[1, n2]
                edge_len += np.sqrt(dx * dx + dy * dy)

            # 等效节点力
            for i in range(6):
                F[2 * enodes[i]] += traction[0] * edge_len / 6.0 * self.thickness
                F[2 * enodes[i] + 1] += traction[1] * edge_len / 6.0 * self.thickness

        return F

    def solve(self, bc_nodes: np.ndarray,
              body_force: Optional[Callable] = None,
              traction: Optional[Tuple[np.ndarray, np.ndarray]] = None
              ) -> np.ndarray:
        """
        求解有限元系统。

        Parameters
        ----------
        bc_nodes : np.ndarray
            Dirichlet 边界节点索引
        body_force : callable, optional
            体力函数 f(x, y) -> (fx, fy)
        traction : tuple, optional
            (boundary_elements, traction_vector)

        Returns
        -------
        U : np.ndarray, shape (n_dofs,)
            位移解向量 [u1x, u1y, u2x, u2y, ...]
        """
        K = self._assemble_stiffness()
        F = self._assemble_load(body_force)

        if traction is not None:
            F = self.apply_neumann_bc(F, traction[0], traction[1])

        K, F = self.apply_dirichlet_bc(K, F, bc_nodes)

        # 求解稀疏线性系统
        U = spsolve(K, F)
        return U

    def compute_strain_energy_density(self, U: np.ndarray) -> np.ndarray:
        """
        计算每个单元的应变能密度。

        U_strain = 0.5 * ε^T D ε
        """
        sed = np.zeros(self.n_elements)

        for e in range(self.n_elements):
            enodes = self.element_node[:, e]
            x_phys = self.node_xy[0, enodes]
            y_phys = self.node_xy[1, enodes]

            E_avg = np.mean(self.E_field[enodes])
            D = elastic_matrix_plane_stress(E_avg, self.nu)

            # 在单元中心求应变
            xi_c = 1.0 / 3.0
            eta_c = 1.0 / 3.0
            _, dphi_dx, dphi_dy = t6_physical_derivatives(
                xi_c, eta_c, x_phys, y_phys)

            # 节点位移
            u_e = np.zeros(12)
            for i in range(6):
                u_e[2 * i] = U[2 * enodes[i]]
                u_e[2 * i + 1] = U[2 * enodes[i] + 1]

            # 应变向量 [ε_xx, ε_yy, γ_xy]
            epsilon = np.zeros(3)
            for i in range(6):
                epsilon[0] += dphi_dx[i] * u_e[2 * i]
                epsilon[1] += dphi_dy[i] * u_e[2 * i + 1]
                epsilon[2] += dphi_dy[i] * u_e[2 * i] + dphi_dx[i] * u_e[2 * i + 1]

            sed[e] = 0.5 * np.dot(epsilon, D @ epsilon)

        return sed

    def compute_nodal_stress(self, U: np.ndarray) -> np.ndarray:
        """
        计算节点应力 [σ_xx, σ_yy, τ_xy]。
        """
        nodal_stress = np.zeros((self.n_nodes, 3))
        nodal_count = np.zeros(self.n_nodes)

        for e in range(self.n_elements):
            enodes = self.element_node[:, e]
            x_phys = self.node_xy[0, enodes]
            y_phys = self.node_xy[1, enodes]

            E_avg = np.mean(self.E_field[enodes])
            D = elastic_matrix_plane_stress(E_avg, self.nu)

            xi_c = 1.0 / 3.0
            eta_c = 1.0 / 3.0
            _, dphi_dx, dphi_dy = t6_physical_derivatives(
                xi_c, eta_c, x_phys, y_phys)

            u_e = np.zeros(12)
            for i in range(6):
                u_e[2 * i] = U[2 * enodes[i]]
                u_e[2 * i + 1] = U[2 * enodes[i] + 1]

            epsilon = np.zeros(3)
            for i in range(6):
                epsilon[0] += dphi_dx[i] * u_e[2 * i]
                epsilon[1] += dphi_dy[i] * u_e[2 * i + 1]
                epsilon[2] += dphi_dy[i] * u_e[2 * i] + dphi_dx[i] * u_e[2 * i + 1]

            sigma = D @ epsilon
            for i in range(6):
                nodal_stress[enodes[i], :] += sigma
                nodal_count[enodes[i]] += 1

        for n in range(self.n_nodes):
            if nodal_count[n] > 0:
                nodal_stress[n, :] /= nodal_count[n]

        return nodal_stress
