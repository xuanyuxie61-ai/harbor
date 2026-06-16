"""
fem_discretization.py - 有限元离散化模块
========================================

融合种子项目:
  - 404_fem2d_heat_rectangle: 二维热方程有限元
  - 963_r83_np: 三对角矩阵操作
  - 933_pyramid_integrals: 区域积分公式

核心数学: 反应-扩散方程的有限元离散化

在凸优化框架中, PDE 约束通过有限元离散化为代数约束.

PDE (反应-扩散方程):
  du/dt - D * Delta u = f(u) + g(x) * u_ctrl   in Omega x (0, T)
  u = 0                                          on dOmega x (0, T)
  u(x, 0) = u_0(x)                              in Omega

  其中:
    D = 扩散系数矩阵
    f(u) = 反应项 (如 Fisher 方程的 u(1-u), Gray-Scott 的 uv^2)
    g(x) = 控制分布函数
    u_ctrl = 控制变量 (优化对象)

弱形式 (Galerkin):
  求 u_h in V_h 使得对任意 v_h in V_h:
    (du_h/dt, v_h) + D * (nabla u_h, nabla v_h) = (f(u_h) + g u_ctrl, v_h)

  其中 (.,.) 是 L^2 内积.

  令 u_h = sum_j U_j phi_j(x), 得:
    M dU/dt + K U = F(U) + G u_ctrl

  其中:
    M_ij = (phi_j, phi_i)     质量矩阵
    K_ij = D (nabla phi_j, nabla phi_i)  刚度矩阵
    F_j = (f(u_h), phi_j)     反应向量
    G_j = (g phi_j, phi_i)    控制耦合矩阵

向后 Euler 时间离散:
  (M/dt + K) U^{n+1} = M/dt U^n + F(U^n) + G u_ctrl^{n+1}

  记 A = M/dt + K, b^n = M/dt U^n + F(U^n),
  则: A U^{n+1} = b^n + G u_ctrl^{n+1}

  这就是凸优化中的等式约束 A x = b 的形式.

二次基函数 (T6 单元):
  使用 6 节点三角形单元, 基函数为二次多项式:
    phi_1 = 2(1-r-s)(1/2-r-s)   (顶点 1)
    phi_2 = 2r(r-1/2)            (顶点 2)
    phi_3 = 2s(s-1/2)            (顶点 3)
    phi_4 = 4r(1-r-s)            (边中点 1-2)
    phi_5 = 4rs                   (边中点 2-3)
    phi_6 = 4s(1-r-s)            (边中点 3-1)
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Dict, Optional


class FEMDiscretization:
    """
    二维反应-扩散方程的有限元离散化.

    使用 6 节点三角形单元 (T6), 二次 Lagrange 基函数.
    在多边形域 Omega 上生成结构化网格.

    生成的线性系统具有带状结构, 使用三对角压缩存储
    (融合 963_r83_np 的 R83 格式思想).

    Parameters
    ----------
    nx, ny : int
        x, y 方向的网格段数
    domain_vertices : ndarray
        多边形域顶点, shape (N, 2)
    diffusion : float
        扩散系数 D
    """

    def __init__(self, nx: int, ny: int,
                 domain_vertices: np.ndarray,
                 diffusion: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.vertices = domain_vertices
        self.D = diffusion

        # 计算域边界
        self.xl = float(np.min(domain_vertices[:, 0]))
        self.xr = float(np.max(domain_vertices[:, 0]))
        self.yb = float(np.min(domain_vertices[:, 1]))
        self.yt = float(np.max(domain_vertices[:, 1]))

        # 节点和单元计数 (6节点三角形)
        self.nnodes = 6  # 每单元节点数
        self.element_num = (nx - 1) * (ny - 1) * 2
        self.node_num = (2 * nx - 1) * (2 * ny - 1)

        # 生成网格
        self.node_xy = self._generate_nodes()
        self.element_node = self._generate_elements()
        self.node_boundary = self._identify_boundary_nodes()

        # 计算单元面积和积分规则
        self.element_area = self._compute_areas()

        # 半带宽
        self.half_bandwidth = self._compute_bandwidth()

        # 组装全局矩阵
        self._assembled = False
        self.M_global = None
        self.K_global = None

    def _generate_nodes(self) -> np.ndarray:
        """生成节点坐标."""
        nx, ny = self.nx, self.ny
        node_num = (2 * nx - 1) * (2 * ny - 1)
        node_xy = np.zeros((node_num, 2))

        for j in range(2 * ny - 1):
            for i in range(2 * nx - 1):
                idx = j * (2 * nx - 1) + i
                node_xy[idx, 0] = ((2 * nx - i - 1) * self.xl +
                                    i * self.xr) / (2 * nx - 2)
                node_xy[idx, 1] = ((2 * ny - j - 1) * self.yb +
                                    j * self.yt) / (2 * ny - 2)

        return node_xy

    def _generate_elements(self) -> np.ndarray:
        """生成 6 节点三角形单元."""
        nx, ny = self.nx, self.ny
        element_num = self.element_num
        element_node = np.zeros((element_num, 6), dtype=int)

        element = 0
        for j in range(ny - 1):
            for i in range(nx - 1):
                sw = j * 2 * (2 * nx - 1) + 2 * i
                w = sw + 1
                nw = sw + 2
                s = sw + 2 * nx - 1
                c = s + 1
                n = s + 2
                se = s + 2 * nx - 1
                e = se + 1
                ne = se + 2

                # 下三角形
                element_node[element] = [sw, se, nw, s, c, w]
                element += 1
                # 上三角形
                element_node[element] = [ne, nw, se, n, c, e]
                element += 1

        return element_node

    def _identify_boundary_nodes(self) -> np.ndarray:
        """识别边界节点 (1=边界, 0=内部)."""
        nx, ny = self.nx, self.ny
        node_num = self.node_num
        node_boundary = np.zeros(node_num, dtype=int)

        for j in range(2 * ny - 1):
            for i in range(2 * nx - 1):
                idx = j * (2 * nx - 1) + i
                if j == 0 or j == 2 * ny - 2 or i == 0 or i == 2 * nx - 2:
                    node_boundary[idx] = 1

        return node_boundary

    def _compute_areas(self) -> np.ndarray:
        """计算每个三角形的面积."""
        element_area = np.zeros(self.element_num)
        for e in range(self.element_num):
            i1, i2, i3 = self.element_node[e, :3]
            x1, y1 = self.node_xy[i1]
            x2, y2 = self.node_xy[i2]
            x3, y3 = self.node_xy[i3]
            element_area[e] = 0.5 * abs(
                y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2))
        return element_area

    def _compute_bandwidth(self) -> int:
        """计算系数矩阵半带宽."""
        ib = 0
        for e in range(self.element_num):
            for iln in range(self.nnodes):
                i = self.element_node[e, iln]
                for jln in range(self.nnodes):
                    j = self.element_node[e, jln]
                    ib = max(ib, abs(j - i))
        return ib

    def _quadratic_basis(self, x: float, y: float,
                          element: int, inode: int
                          ) -> Tuple[float, float, float]:
        """
        计算二次基函数值及其导数.

        参考坐标系 (R, S) 到物理坐标 (X, Y) 的映射:
          X = X1 + (X2-X1)*R + (X3-X1)*S
          Y = Y1 + (Y2-Y1)*R + (Y3-Y1)*S

        Jacobian:
          det = (X2-X1)(Y3-Y1) - (X3-X1)(Y2-Y1)
          dR/dX = (Y3-Y1)/det,  dR/dY = (X1-X3)/det
          dS/dX = (Y1-Y2)/det,  dS/dY = (X2-X1)/det

        基函数在参考坐标 (R,S) 中的定义:
          phi_1 = 2(1-R-S)(1/2-R-S)
          phi_2 = 2R(R-1/2)
          phi_3 = 2S(S-1/2)
          phi_4 = 4R(1-R-S)
          phi_5 = 4RS
          phi_6 = 4S(1-R-S)

        Parameters
        ----------
        x, y : float
            物理坐标
        element : int
            单元编号
        inode : int
            局部节点编号 (1-6)

        Returns
        -------
        b : float
            基函数值
        dbdx, dbdy : float
            基函数对 x, y 的偏导数
        """
        nodes = self.element_node[element, :3]
        xn = self.node_xy[nodes, 0]
        yn = self.node_xy[nodes, 1]

        # Jacobian 行列式
        det = ((xn[1] - xn[0]) * (yn[2] - yn[0]) -
               (xn[2] - xn[0]) * (yn[1] - yn[0]))

        if abs(det) < 1.0e-30:
            return 0.0, 0.0, 0.0

        # 参考坐标
        r = ((yn[2] - yn[0]) * (x - xn[0]) +
             (xn[0] - xn[2]) * (y - yn[0])) / det
        s = ((yn[0] - yn[1]) * (x - xn[0]) +
             (xn[1] - xn[0]) * (y - yn[0])) / det

        # 坐标映射导数
        drdx = (yn[2] - yn[0]) / det
        drdy = (xn[0] - xn[2]) / det
        dsdx = (yn[0] - yn[1]) / det
        dsdy = (xn[1] - xn[0]) / det

        # 基函数值及参考坐标导数
        if inode == 0:
            b = 2.0 * (1.0 - r - s) * (0.5 - r - s)
            dbdr = -3.0 + 4.0 * r + 4.0 * s
            dbds = -3.0 + 4.0 * r + 4.0 * s
        elif inode == 1:
            b = 2.0 * r * (r - 0.5)
            dbdr = -1.0 + 4.0 * r
            dbds = 0.0
        elif inode == 2:
            b = 2.0 * s * (s - 0.5)
            dbdr = 0.0
            dbds = -1.0 + 4.0 * s
        elif inode == 3:
            b = 4.0 * r * (1.0 - r - s)
            dbdr = 4.0 - 8.0 * r - 4.0 * s
            dbds = -4.0 * r
        elif inode == 4:
            b = 4.0 * r * s
            dbdr = 4.0 * s
            dbds = 4.0 * r
        elif inode == 5:
            b = 4.0 * s * (1.0 - r - s)
            dbdr = -4.0 * s
            dbds = 4.0 - 4.0 * r - 8.0 * s
        else:
            return 0.0, 0.0, 0.0

        # 链式法则转换到物理坐标
        dbdx = dbdr * drdx + dbds * dsdx
        dbdy = dbdr * drdy + dbds * dsdy

        return b, dbdx, dbdy

    def assemble_system(self, dt: float) -> Tuple[sparse.csc_matrix,
                                                    sparse.csc_matrix]:
        """
        组装全局质量矩阵 M 和刚度矩阵 K.

        质量矩阵: M_ij = integral phi_i * phi_j dx dy
        刚度矩阵: K_ij = D * integral nabla phi_i . nabla phi_j dx dy

        使用 3 点 Gauss 积分规则 (对三角形足够精确).

        3 点规则 (重心坐标):
          点 1: (1/6, 1/6, 2/3),  权 = 1/3
          点 2: (1/6, 2/3, 1/6),  权 = 1/3
          点 3: (2/3, 1/6, 1/6),  权 = 1/3

        注意: 这些点是边中点, 对二次多项式精确.

        Parameters
        ----------
        dt : float
            时间步长

        Returns
        -------
        M : sparse matrix
            质量矩阵
        K : sparse matrix
            刚度矩阵
        """
        n = self.node_num

        # 3 点 Gauss 规则 (边中点)
        wq = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        # 组装
        rows = []
        cols = []
        m_data = []
        k_data = []

        for e in range(self.element_num):
            # 3 个顶点
            v_nodes = self.element_node[e, :3]
            x1, y1 = self.node_xy[v_nodes[0]]
            x2, y2 = self.node_xy[v_nodes[1]]
            x3, y3 = self.node_xy[v_nodes[2]]

            # 3 个积分点 (边中点)
            xq = np.array([0.5 * (x1 + x2), 0.5 * (x2 + x3),
                           0.5 * (x1 + x3)])
            yq = np.array([0.5 * (y1 + y2), 0.5 * (y2 + y3),
                           0.5 * (y1 + y3)])

            for q in range(3):
                xqp, yqp = xq[q], yq[q]
                w = self.element_area[e] * wq[q]

                for test in range(self.nnodes):
                    i = self.element_node[e, test]
                    bi, dbidx, dbidy = self._quadratic_basis(
                        xqp, yqp, e, test)

                    for basis in range(self.nnodes):
                        j = self.element_node[e, basis]
                        bj, dbjdx, dbjdy = self._quadratic_basis(
                            xqp, yqp, e, basis)

                        # 质量矩阵贡献
                        mij = bi * bj
                        # 刚度矩阵贡献
                        kij = self.D * (dbidx * dbjdx + dbidy * dbjdy)

                        rows.append(i)
                        cols.append(j)
                        m_data.append(w * mij)
                        k_data.append(w * kij)

        M = sparse.coo_matrix(
            (m_data, (rows, cols)), shape=(n, n)).tocsc()
        K = sparse.coo_matrix(
            (k_data, (rows, cols)), shape=(n, n)).tocsc()

        self.M_global = M
        self.K_global = K
        self._assembled = True

        return M, K

    def build_time_step_matrix(self, dt: float) -> sparse.csc_matrix:
        """
        构造向后 Euler 的系统矩阵 A = M/dt + K.

        向后 Euler 格式:
          M (U^{n+1} - U^n) / dt + K U^{n+1} = F^{n+1}
          => (M/dt + K) U^{n+1} = M/dt U^n + F^{n+1}

        矩阵 A = M/dt + K 是 SPD 的 (对称正定),
        这保证了 Newton 系统的可解性.

        Parameters
        ----------
        dt : float
            时间步长

        Returns
        -------
        sparse matrix
            系统矩阵 A = M/dt + K
        """
        if not self._assembled:
            self.assemble_system(dt)

        A = (1.0 / dt) * self.M_global + self.K_global
        return A

    def apply_dirichlet_bc(self, A: sparse.csc_matrix,
                            rhs: np.ndarray, time: float,
                            bc_func=None) -> Tuple[sparse.csc_matrix, np.ndarray]:
        """
        施加 Dirichlet 边界条件.

        对边界节点 i:
          A[i, :] = 0,  A[i, i] = 1
          rhs[i] = g(x_i, y_i, time)

        这保持了矩阵的稀疏性, 同时精确施加边界值.

        Parameters
        ----------
        A : sparse matrix
            系统矩阵
        rhs : ndarray
            右端向量
        time : float
            当前时间
        bc_func : callable, optional
            边界条件函数 bc_func(x, y, t) -> value

        Returns
        -------
        A_modified : sparse matrix
            修改后的矩阵
        rhs_modified : ndarray
            修改后的右端
        """
        A_mod = A.tolil()
        rhs_mod = rhs.copy()

        if bc_func is None:
            bc_func = lambda x, y, t: 0.0

        for i in range(self.node_num):
            if self.node_boundary[i] == 1:
                x, y = self.node_xy[i]
                bc_val = bc_func(x, y, time)

                # 清零该行
                A_mod[i, :] = 0.0
                A_mod[i, i] = 1.0
                rhs_mod[i] = bc_val

        return A_mod.tocsc(), rhs_mod

    def get_tridiagonal_approximation(self, A: sparse.csc_matrix
                                       ) -> Tuple[np.ndarray, np.ndarray,
                                                   np.ndarray]:
        """
        提取三对角近似 (融合 963_r83_np 的 R83 存储).

        在带状矩阵中, 主对角线和相邻对角线包含主要信息.
        R83 存储:
          a[0, :] = 超对角线
          a[1, :] = 主对角线
          a[2, :] = 次对角线

        这用于预处理和快速近似求解.

        Parameters
        ----------
        A : sparse matrix
            系统矩阵

        Returns
        -------
        r83 : ndarray, shape (3, n)
            三对角存储
        """
        n = self.node_num
        A_dense = A.toarray() if not isinstance(A, np.ndarray) else A
        r83 = np.zeros((3, n))

        # 主对角线
        r83[1, :] = np.diag(A_dense)
        # 超对角线
        r83[0, 1:] = np.diag(A_dense, 1)
        # 次对角线
        r83[2, :-1] = np.diag(A_dense, -1)

        return r83

    def tridiagonal_solve(self, r83: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        三对角系统求解 (Thomas 算法, 融合 963_r83_np).

        Thomas 算法 (追赶法):
          前代: 消去次对角线
            m_i = a_{i,i-1} / a_{i-1,i-1}
            a_{i,i} -= m_i * a_{i-1,i}
            b_i -= m_i * b_{i-1}
          回代:
            x_n = b_n / a_{n,n}
            x_i = (b_i - a_{i,i+1} * x_{i+1}) / a_{i,i}

        复杂度: O(n), 远优于一般 LU 分解的 O(n^3).

        Parameters
        ----------
        r83 : ndarray, shape (3, n)
            三对角存储
        b : ndarray
            右端向量

        Returns
        -------
        x : ndarray
            解向量
        """
        n = len(b)
        # 复制避免修改
        diag = r83[1, :].copy()
        upper = r83[0, :].copy()
        lower = r83[2, :].copy()
        rhs = b.copy()

        # 前代
        for i in range(1, n):
            if abs(diag[i - 1]) < 1.0e-30:
                diag[i - 1] = 1.0e-30
            m = lower[i - 1] / diag[i - 1]
            diag[i] -= m * upper[i - 1]
            rhs[i] -= m * rhs[i - 1]

        # 回代
        x = np.zeros(n)
        if abs(diag[n - 1]) < 1.0e-30:
            diag[n - 1] = 1.0e-30
        x[n - 1] = rhs[n - 1] / diag[n - 1]

        for i in range(n - 2, -1, -1):
            if abs(diag[i]) < 1.0e-30:
                diag[i] = 1.0e-30
            x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i]

        return x

    def compute_pyrmaid_integral(self, expon: Tuple[int, int, int]) -> float:
        """
        锥体区域上的单项式积分 (融合 933_pyramid_integrals).

        在优化目标中, 需要计算区域上的积分:
          I = integral_Omega x^a y^b z^c dx dy dz

        对单位锥体:
          -(1-z) <= x <= 1-z
          -(1-z) <= y <= 1-z
          0 <= z <= 1

        解析公式 (Stroud, 1971):
          若 a, b 均为偶数:
            I = 2/(a+1) * 2/(b+1) * sum_{i=0}^{a+b+2} (-1)^i C(a+b+2,i) / (i+c+1)
          否则: I = 0 (对称性)

        Parameters
        ----------
        expon : tuple of 3 ints
            (a, b, c) 指数

        Returns
        -------
        float
            积分值
        """
        a, b, c = expon

        # 对称性: 若 a 或 b 为奇数, 积分为 0
        if a % 2 != 0 or b % 2 != 0:
            return 0.0

        i_hi = 2 + a + b
        value = 0.0
        for i in range(i_hi + 1):
            sign = (-1) ** i
            from math import comb
            value += sign * comb(i_hi, i) / (i + c + 1)

        value *= 2.0 / (a + 1) * 2.0 / (b + 1)
        return value
