"""
fem_2d_serene.py
================
二维 serendipity 有限元求解器

基于种子项目 402_fem2d_bvp_serene，将 MATLAB 的 serendipity 基函数
有限元方法迁移至 Python，用于求解二维对流-扩散-反应方程：

    -∇·(a(x,y) ∇u) + c(x,y) u = f(x,y)

采用 8 节点 serendipity 四边形单元：
    节点编号：
      3 -- 2 -- 1
      |          |
      4          8
      |          |
      5 -- 6 -- 7

每个单元使用 3×3 Gauss-Legendre 数值积分。
"""

import numpy as np


class SerendipityElement:
    """
    8 节点 serendipity 四边形单元。
    """

    def __init__(self):
        """
        初始化 Gauss-Legendre 求积点和权重（3点公式）。
        """
        # 3点 Gauss-Legendre 求积
        self.quad_xi = np.array([
            -0.7745966692414834,
             0.0,
             0.7745966692414834
        ])
        self.quad_w = np.array([
            0.5555555555555556,
            0.8888888888888889,
            0.5555555555555556
        ])
        self.quad_num = 3

    def basis_functions(self, xi, eta):
        """
        计算 serendipity 基函数在参考坐标 (ξ, η) ∈ [-1,1]² 处的值。

        8 节点基函数：
            N1 = (1+ξ)(1+η)(ξ+η-1) / 4
            N2 = (1-ξ²)(1+η) / 2
            N3 = (1-ξ)(1+η)(-ξ+η-1) / 4
            N4 = (1-ξ)(1-η²) / 2
            N5 = (1-ξ)(1-η)(-ξ-η-1) / 4
            N6 = (1-ξ²)(1-η) / 2
            N7 = (1+ξ)(1-η)(ξ-η-1) / 4
            N8 = (1+ξ)(1-η²) / 2

        Parameters
        ----------
        xi, eta : float
            参考坐标。

        Returns
        -------
        ndarray, shape (8,)
            8 个基函数值。
        """
        N = np.zeros(8)
        N[0] = (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0) * 0.25
        N[1] = (1.0 - xi ** 2) * (1.0 + eta) * 0.5
        N[2] = (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0) * 0.25
        N[3] = (1.0 - xi) * (1.0 - eta ** 2) * 0.5
        N[4] = (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0) * 0.25
        N[5] = (1.0 - xi ** 2) * (1.0 - eta) * 0.5
        N[6] = (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0) * 0.25
        N[7] = (1.0 + xi) * (1.0 - eta ** 2) * 0.5
        return N

    def basis_derivatives(self, xi, eta):
        """
        计算 serendipity 基函数在参考坐标下的偏导数 ∂N/∂ξ, ∂N/∂η。

        Parameters
        ----------
        xi, eta : float
            参考坐标。

        Returns
        -------
        tuple of ndarray
            (dN_dxi, dN_deta)，每个 shape 为 (8,)。
        """
        dN_dxi = np.zeros(8)
        dN_deta = np.zeros(8)

        dN_dxi[0] = (1.0 + eta) * (2.0 * xi + eta) * 0.25
        dN_dxi[1] = -xi * (1.0 + eta)
        dN_dxi[2] = (1.0 + eta) * (2.0 * xi - eta) * (-0.25)
        dN_dxi[3] = -(1.0 - eta ** 2) * 0.5
        dN_dxi[4] = (1.0 - eta) * (2.0 * xi + eta) * (-0.25)
        dN_dxi[5] = -xi * (1.0 - eta)
        dN_dxi[6] = (1.0 - eta) * (2.0 * xi - eta) * 0.25
        dN_dxi[7] = (1.0 - eta ** 2) * 0.5

        dN_deta[0] = (1.0 + xi) * (xi + 2.0 * eta) * 0.25
        dN_deta[1] = (1.0 - xi ** 2) * 0.5
        dN_deta[2] = (1.0 - xi) * (-xi + 2.0 * eta) * 0.25
        dN_deta[3] = (1.0 - xi) * (-eta)
        dN_deta[4] = (1.0 - xi) * (-xi - 2.0 * eta) * (-0.25)
        dN_deta[5] = (1.0 - xi ** 2) * (-0.5)
        dN_deta[6] = (1.0 + xi) * (xi - 2.0 * eta) * (-0.25)
        dN_deta[7] = (1.0 + xi) * (-eta)

        return dN_dxi, dN_deta

    def jacobian_and_derivatives(self, xi, eta, x_nodes, y_nodes):
        """
        计算 Jacobian 矩阵及其行列式，以及全局坐标下的基函数导数。

        Jacobian：
            J = [∂x/∂ξ  ∂y/∂ξ]
                [∂x/∂η  ∂y/∂η]

        det(J) = (∂x/∂ξ)(∂y/∂η) - (∂x/∂η)(∂y/∂ξ)

        全局导数：
            [∂N/∂x]   =  J^{-1} [∂N/∂ξ]
            [∂N/∂y]             [∂N/∂η]

        Parameters
        ----------
        xi, eta : float
            参考坐标。
        x_nodes, y_nodes : ndarray, shape (8,)
            单元节点坐标。

        Returns
        -------
        tuple
            (detJ, dN_dx, dN_dy)
        """
        dN_dxi, dN_deta = self.basis_derivatives(xi, eta)

        dx_dxi = np.dot(dN_dxi, x_nodes)
        dy_dxi = np.dot(dN_dxi, y_nodes)
        dx_deta = np.dot(dN_deta, x_nodes)
        dy_deta = np.dot(dN_deta, y_nodes)

        detJ = dx_dxi * dy_deta - dx_deta * dy_dxi

        if abs(detJ) < 1e-14:
            raise ValueError("Jacobian 行列式接近零，单元退化")

        inv_detJ = 1.0 / detJ
        dN_dx = inv_detJ * (dN_dxi * dy_deta - dN_deta * dy_dxi)
        dN_dy = inv_detJ * (-dN_dxi * dx_deta + dN_deta * dx_dxi)

        return detJ, dN_dx, dN_dy


class FEM2DSerene:
    """
    基于 serendipity 单元的二维有限元求解器。
    """

    def __init__(self, nx, ny, x_coords, y_coords):
        """
        初始化 FEM 网格。

        Parameters
        ----------
        nx, ny : int
            x 和 y 方向的节点数，必须为奇数且 ≥ 3。
        x_coords : ndarray, shape (nx,)
            x 方向节点坐标。
        y_coords : ndarray, shape (ny,)
            y 方向节点坐标。
        """
        if nx % 2 == 0 or ny % 2 == 0:
            raise ValueError("nx 和 ny 必须为奇数")
        if nx < 3 or ny < 3:
            raise ValueError("nx 和 ny 必须至少为 3")

        self.nx = nx
        self.ny = ny
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.element = SerendipityElement()

        # 单元数
        self.ex_num = (nx - 1) // 2
        self.ey_num = (ny - 1) // 2

        # 总节点数（serendipity 网格的节点编号）
        self.num_nodes = self._compute_node_num()

    def _compute_node_num(self):
        """
        计算 serendipity 网格的总节点数。

        对于 nx × ny 的底层网格（nx, ny 为奇数），
        serendipity 单元每 2×2 底层网格形成一个 8 节点单元，
        但节点共享。
        """
        # 简化：直接使用底层网格点作为节点（将 serendipity 退化为双线性）
        # 或保持 serendipity 节点结构
        # 这里采用完整网格点数
        return self.nx * self.ny

    def _global_to_local_node_map(self, ex, ey):
        """
        将全局网格索引映射为 serendipity 单元的 8 个局部节点。

        底层网格索引：
            ex 从 0 到 ex_num-1，每个占 2 个网格步
            ey 从 0 到 ey_num-1，每个占 2 个网格步

        局部节点编号（按 serendipity 顺序）：
            3 -- 2 -- 1
            |          |
            4          8
            |          |
            5 -- 6 -- 7
        """
        i0 = 2 * ex
        j0 = 2 * ey

        # 8 个节点的 (i, j) 索引
        local_indices = [
            (i0 + 2, j0 + 2),  # 1: 右上
            (i0 + 1, j0 + 2),  # 2: 上中
            (i0, j0 + 2),      # 3: 左上
            (i0, j0 + 1),      # 4: 左中
            (i0, j0),          # 5: 左下
            (i0 + 1, j0),      # 6: 下中
            (i0 + 2, j0),      # 7: 右下
            (i0 + 2, j0 + 1),  # 8: 右中
        ]

        # 转换为全局线性索引
        node_ids = []
        for i, j in local_indices:
            if 0 <= i < self.nx and 0 <= j < self.ny:
                node_ids.append(i * self.ny + j)
            else:
                raise IndexError("节点索引越界")

        return np.array(node_ids, dtype=int)

    def assemble_system(self, a_func, c_func, f_func):
        """
        组装有限元全局刚度矩阵和载荷向量。

        求解方程：-∇·(a∇u) + c u = f

        弱形式：∫ a ∇u·∇v dΩ + ∫ c u v dΩ = ∫ f v dΩ

        Parameters
        ----------
        a_func : callable
            扩散系数 a(x, y)。
        c_func : callable
            反应系数 c(x, y)。
        f_func : callable
            右端项 f(x, y)。

        Returns
        -------
        tuple
            (K, F) 全局刚度矩阵和载荷向量。
        """
        K = np.zeros((self.num_nodes, self.num_nodes))
        F = np.zeros(self.num_nodes)

        for ey in range(self.ey_num):
            for ex in range(self.ex_num):
                # 获取单元节点坐标
                node_ids = self._global_to_local_node_map(ex, ey)
                x_nodes = np.array([self.x_coords[i // self.ny] for i in node_ids])
                y_nodes = np.array([self.y_coords[i % self.ny] for i in node_ids])

                # 单元刚度矩阵和载荷向量
                Ke = np.zeros((8, 8))
                Fe = np.zeros(8)

                for qx in range(self.element.quad_num):
                    xi = self.element.quad_xi[qx]
                    wx = self.element.quad_w[qx]
                    for qy in range(self.element.quad_num):
                        eta = self.element.quad_xi[qy]
                        wy = self.element.quad_w[qy]

                        detJ, dN_dx, dN_dy = self.element.jacobian_and_derivatives(
                            xi, eta, x_nodes, y_nodes
                        )
                        N = self.element.basis_functions(xi, eta)

                        # 物理坐标
                        xq = np.dot(N, x_nodes)
                        yq = np.dot(N, y_nodes)

                        aq = a_func(xq, yq)
                        cq = c_func(xq, yq)
                        fq = f_func(xq, yq)

                        wq = wx * wy * abs(detJ)

                        for i in range(8):
                            for j in range(8):
                                Ke[i, j] += wq * (
                                    aq * (dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j]) +
                                    cq * N[i] * N[j]
                                )
                            Fe[i] += wq * fq * N[i]

                # 组装到全局矩阵
                for i in range(8):
                    ii = node_ids[i]
                    for j in range(8):
                        jj = node_ids[j]
                        K[ii, jj] += Ke[i, j]
                    F[ii] += Fe[i]

        return K, F

    def apply_dirichlet_bc(self, K, F, bc_nodes, bc_values):
        """
        施加 Dirichlet 边界条件。

        对于边界节点 k，将其对应行和列置零，对角元置 1，
        右端项设为边界值。

        Parameters
        ----------
        K : ndarray
            全局刚度矩阵。
        F : ndarray
            全局载荷向量。
        bc_nodes : list of int
            边界节点索引。
        bc_values : list of float
            边界节点值。

        Returns
        -------
        tuple
            (K, F) 修改后的矩阵和向量。
        """
        K = K.copy()
        F = F.copy()

        for node, val in zip(bc_nodes, bc_values):
            K[node, :] = 0.0
            K[:, node] = 0.0
            K[node, node] = 1.0
            F[node] = val

        return K, F

    def solve(self, K, F):
        """
        求解线性系统 K u = F。

        Parameters
        ----------
        K : ndarray
            刚度矩阵。
        F : ndarray
            载荷向量。

        Returns
        -------
        ndarray
            解向量 u。
        """
        return np.linalg.solve(K, F)

    def compute_l2_error(self, u, exact_func):
        """
        计算有限元解的 L2 误差：
            ||u - u_exact||_L2 = sqrt(∫ (u - u_exact)² dΩ)

        Parameters
        ----------
        u : ndarray
            有限元解。
        exact_func : callable
            精确解函数 exact_func(x, y)。

        Returns
        -------
        float
            L2 误差。
        """
        error_sq = 0.0

        for ey in range(self.ey_num):
            for ex in range(self.ex_num):
                node_ids = self._global_to_local_node_map(ex, ey)
                x_nodes = np.array([self.x_coords[i // self.ny] for i in node_ids])
                y_nodes = np.array([self.y_coords[i % self.ny] for i in node_ids])
                u_nodes = u[node_ids]

                for qx in range(self.element.quad_num):
                    xi = self.element.quad_xi[qx]
                    wx = self.element.quad_w[qx]
                    for qy in range(self.element.quad_num):
                        eta = self.element.quad_xi[qy]
                        wy = self.element.quad_w[qy]

                        detJ, _, _ = self.element.jacobian_and_derivatives(
                            xi, eta, x_nodes, y_nodes
                        )
                        N = self.element.basis_functions(xi, eta)

                        xq = np.dot(N, x_nodes)
                        yq = np.dot(N, y_nodes)
                        uq = np.dot(N, u_nodes)

                        eq = exact_func(xq, yq)
                        wq = wx * wy * abs(detJ)
                        error_sq += wq * (uq - eq) ** 2

        return np.sqrt(error_sq)
