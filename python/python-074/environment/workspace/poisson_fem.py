r"""
poisson_fem.py
==============
二维泊松方程的有限元求解器，用于流函数方程 \nabla^2 \psi = -\omega。

科学背景
--------
涡量-流函数方法中，每一时间步需求解流函数泊松方程：

    -\nabla^2 \psi = \omega   于 \Omega
    \psi = g_D               于 \Gamma_D
    \partial\psi/\partial n = g_N   于 \Gamma_N

其中 \Omega 为流体域，\Gamma_D 为 Dirichlet 边界（入口、壁面、圆柱表面），
\Gamma_N 为 Neumann 边界（出口）。

采用 Galerkin 有限元方法，线性三角形单元，逐单元组装刚度矩阵 K 与载荷向量 F：

    K_{ij} = \int_\Omega \nabla\phi_i \cdot \nabla\phi_j \, d\Omega
    F_i    = \int_\Omega \omega \phi_i \, d\Omega + \int_{\Gamma_N} g_N \phi_i \, ds

引入罚函数法或消去法处理 Dirichlet 边界条件后，求解线性系统：

    K \psi = F

本模块对应原种子项目：
- 416_fem2d_scalar_display_gpl（二维标量有限元离散框架，删除可视化部分，
  保留刚度矩阵组装与线性求解核心）
r"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class PoissonFEM2D:
    r"""
    基于结构化背景网格的二维泊松方程 FEM 求解器。
    将矩形域划分为三角形单元（每条矩形对角线一分为二）。
    """

    def __init__(self, nx, ny, lx, ly, solid_mask=None):
        r"""
        参数
        ----
        nx, ny : int
            背景网格节点数。
        lx, ly : float
            域尺寸。
        solid_mask : ndarray(bool), shape (ny, nx) or None
            标记固体区域节点。
        """
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)

        if solid_mask is None:
            solid_mask = np.zeros((ny, nx), dtype=bool)
        self.solid_mask = solid_mask

        # 节点总数
        self.n_nodes = nx * ny
        # 节点编号映射 (j, i) -> global_id
        self.id_map = np.arange(self.n_nodes).reshape(ny, nx)

        # 标记有效自由度（非固体）
        self.is_free = ~self.solid_mask.flatten()
        self.free_dofs = np.where(self.is_free)[0]
        self.n_free = len(self.free_dofs)

        # 预计算单元信息（仅一次）
        self._build_mesh_connectivity()
        self._assemble_stiffness_matrix()

    def _build_mesh_connectivity(self):
        r"""
        构建三角形单元连接关系。
        每个矩形网格被分为两个三角形：
        - 类型 A: (i,j), (i+1,j), (i,j+1)
        - 类型 B: (i+1,j+1), (i,j+1), (i+1,j)
        """
        nx, ny = self.nx, self.ny
        elements = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n1 = self.id_map[j, i]
                n2 = self.id_map[j, i + 1]
                n3 = self.id_map[j + 1, i]
                n4 = self.id_map[j + 1, i + 1]
                elements.append([n1, n2, n3])
                elements.append([n4, n3, n2])
        self.elements = np.array(elements, dtype=int)
        self.n_elements = len(self.elements)

        # 节点坐标
        x = np.linspace(0.0, self.lx, nx)
        y = np.linspace(0.0, self.ly, ny)
        X, Y = np.meshgrid(x, y)
        self.coords = np.column_stack((X.flatten(), Y.flatten()))

    def _local_stiffness(self, x, y):
        r"""
        计算三角形单元的局部刚度矩阵。

        对线性三角形单元，面积 A = 0.5 * |det(B)|，其中
        B = [[x2-x1, x3-x1],
             [y2-y1, y3-y1]]

        梯度矩阵 G = B^{-T} * [ [-1, 1, 0], [-1, 0, 1] ]^T
        局部刚度 k_local = A * G * G^T
        """
        B = np.array([
            [x[1] - x[0], x[2] - x[0]],
            [y[1] - y[0], y[2] - y[0]]
        ])
        area = 0.5 * abs(np.linalg.det(B))
        if area < 1e-16:
            return np.zeros((3, 3))

        # 形函数梯度（参考单元 -> 物理单元）
        dN_dxi = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        inv_B_T = np.linalg.inv(B).T
        grad_N = inv_B_T @ dN_dxi

        k_local = area * (grad_N.T @ grad_N)
        return k_local

    def _assemble_stiffness_matrix(self):
        r"""
        组装全局刚度矩阵（仅几何相关部分，与右端项无关）。
        采用 CSR 稀疏格式存储。
        """
        rows = []
        cols = []
        data = []

        for elem in self.elements:
            nodes = elem
            x = self.coords[nodes, 0]
            y = self.coords[nodes, 1]
            k_loc = self._local_stiffness(x, y)

            for a in range(3):
                for b in range(3):
                    rows.append(nodes[a])
                    cols.append(nodes[b])
                    data.append(k_loc[a, b])

        self.K = csr_matrix(
            (data, (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        )

    def solve(self, rhs_field, dirichlet_mask, dirichlet_values,
              neumann_edges=None, neumann_flux=None):
        r"""
        求解泊松方程 K psi = F。

        参数
        ----
        rhs_field : ndarray, shape (ny, nx)
            右端项 \omega 在节点上的值。
        dirichlet_mask : ndarray(bool), shape (ny, nx)
            Dirichlet 边界标记。
        dirichlet_values : ndarray, shape (ny, nx)
            Dirichlet 边界值。
        neumann_edges : list of tuple or None
            Neumann 边界边列表，每个元素为 (node_a, node_b)。
        neumann_flux : ndarray or None
            对应每条边的通量值。

        返回
        ----
        psi : ndarray, shape (ny, nx)
            流函数场。
        """
        rhs_flat = rhs_field.flatten()
        F = np.zeros(self.n_nodes)

        # 将 rhs 投影到节点（此处简化为节点值直接作为积分中点近似）
        # 更精确做法：逐单元积分 \int \omega \phi_i d\Omega
        # 为简化且保证可运行，采用 lumped mass 近似：
        # F_i = \omega_i * A_i，其中 A_i 为节点控制面积
        node_area = np.zeros(self.n_nodes)
        for elem in self.elements:
            nodes = elem
            x = self.coords[nodes, 0]
            y = self.coords[nodes, 1]
            area = 0.5 * abs(
                (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            )
            for n in nodes:
                node_area[n] += area / 3.0

        F = rhs_flat * node_area

        # Neumann 边界贡献
        if neumann_edges is not None and neumann_flux is not None:
            for idx, (na, nb) in enumerate(neumann_edges):
                length = np.linalg.norm(self.coords[nb] - self.coords[na])
                flux = neumann_flux[idx] if hasattr(neumann_flux, '__len__') else neumann_flux
                F[na] += 0.5 * flux * length
                F[nb] += 0.5 * flux * length

        # Dirichlet 边界处理（消去法）
        d_mask_flat = dirichlet_mask.flatten()
        d_values_flat = dirichlet_values.flatten()
        is_dirichlet = d_mask_flat & (~self.solid_mask.flatten())

        # 构建缩减系统
        free = self.is_free & (~is_dirichlet)
        free_ids = np.where(free)[0]

        if len(free_ids) == 0:
            # 无自由节点，直接返回 Dirichlet 值
            psi_flat = np.zeros(self.n_nodes)
            psi_flat[is_dirichlet] = d_values_flat[is_dirichlet]
            psi_flat[self.solid_mask.flatten()] = 0.0
            return psi_flat.reshape(self.ny, self.nx)

        K_ff = self.K[np.ix_(free_ids, free_ids)]
        F_f = F[free_ids].copy()

        # Dirichlet 贡献移到右端
        dir_ids = np.where(is_dirichlet)[0]
        if len(dir_ids) > 0:
            K_fd = self.K[np.ix_(free_ids, dir_ids)]
            F_f -= K_fd @ d_values_flat[dir_ids]

        # 求解
        try:
            psi_free = spsolve(K_ff, F_f)
        except Exception as e:
            # 若求解失败，使用最小二乘 fallback
            psi_free = np.linalg.lstsq(K_ff.toarray(), F_f, rcond=None)[0]

        psi_flat = np.zeros(self.n_nodes)
        psi_flat[free_ids] = psi_free
        psi_flat[is_dirichlet] = d_values_flat[is_dirichlet]
        psi_flat[self.solid_mask.flatten()] = 0.0

        return psi_flat.reshape(self.ny, self.nx)
