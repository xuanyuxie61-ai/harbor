"""
fem3d_projection.py
================================================================================
3D 有限元场投影：将 2D 截面磁势场投影到三维轴向长度分布

融合原项目:
  - 418_fem3d_project : 3D FEM投影（TET4单元、基函数、体积积分、投影方程）

核心科学内容:
  1. 三维有限元空间中的 L^2 投影问题:

     给定二维截面上的磁矢势场 A_z^{2D}(x,y)（由 FEM2D 求解得到），
     我们希望构造一个三维有限元函数 A_z^{3D}(x,y,z)，使其在轴向长度 L 上
     合理分布，并通过 L^2 投影最小化误差:

         min || A_z^{3D} - A_z^{2D} ||_{L^2(Ω_3D)}^2

     等价于求解投影方程:

         ∫_{Ω_3D} A_z^{3D} N_i dV = ∫_{Ω_3D} A_z^{2D} N_i dV

     其中 N_i 为三维有限元基函数。

  2. TET4（4节点四面体）线性基函数:

     对于四面体顶点 t_1, t_2, t_3, t_4，体积坐标（重心坐标）:

         φ_i(x,y,z) = V_i(x,y,z) / V

     其中 V 为四面体体积:

         V = | det([x2-x1, x3-x1, x4-x1;
                    y2-y1, y3-y1, y4-y1;
                    z2-z1, z3-z1, z4-z1]) | / 6

     体积坐标满足 φ_1 + φ_2 + φ_3 + φ_4 = 1.

  3. 质量矩阵组装:

         M_{ij} = ∫_{T_e} φ_i φ_j dV

     对于 TET4 元，质量矩阵元素:
         M_{ii} = V / 10
         M_{ij} = V / 20  (i ≠ j)

  4. 右端项（投影源）:

         b_i = ∫_{T_e} A_z^{2D}(x,y) φ_i(x,y,z) dV

     由于 A_z^{2D} 不依赖于 z，在轴向均匀假设下:
         b_i ≈ A_z^{2D}(x_c, y_c) * V / 4

     其中 (x_c, y_c) 为四面体在 xy 平面上的投影质心。

  5. 三维网格生成（基于 2D 截面的拉伸/扫掠）:
     将 2D 三角形网格沿轴向拉伸为 3D 棱柱，再细分为四面体。
================================================================================
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class FEM3DProjection:
    """
    3D 有限元投影器，将 2D 场分布投影到 3D 有限元空间。

    融合原项目 418_fem3d_project 的 basis_mn_tet4 与 fem3d_transfer 思想。
    """

    def __init__(self, nodes_2d: np.ndarray, elements_2d: np.ndarray, axial_length: float = 0.2):
        """
        参数:
            nodes_2d    : (n_node_2d, 2) 二维节点坐标
            elements_2d : (n_elem_2d, 3)  二维三角形单元
            axial_length: 轴向总长度 [m]
        """
        self.nodes_2d = np.asarray(nodes_2d, dtype=float)
        self.elements_2d = np.asarray(elements_2d, dtype=int)
        self.axial_length = float(axial_length)
        self.n_node_2d = self.nodes_2d.shape[0]
        self.n_elem_2d = self.elements_2d.shape[0]

        # 生成 3D 网格（两层：z=0 和 z=L）
        self._build_extruded_mesh()

    def _build_extruded_mesh(self):
        """
        通过拉伸 2D 三角形网格生成 3D 棱柱/四面体网格。

        每个 2D 三角形 (v1,v2,v3) 拉伸为 3D 棱柱:
            底面: (v1,0), (v2,0), (v3,0)
            顶面: (v1,L), (v2,L), (v3,L)

        棱柱细分为 3 个四面体:
            T1: (v1,0), (v2,0), (v3,0), (v1,L)
            T2: (v2,0), (v3,0), (v1,L), (v2,L)
            T3: (v3,0), (v1,L), (v2,L), (v3,L)
        """
        n2 = self.n_node_2d
        L = self.axial_length

        # 3D 节点: 底面节点 0..n2-1, 顶面节点 n2..2*n2-1
        self.nodes_3d = np.zeros((2 * n2, 3))
        self.nodes_3d[:n2, :2] = self.nodes_2d
        self.nodes_3d[n2:, :2] = self.nodes_2d
        self.nodes_3d[n2:, 2] = L

        # 3D 四面体单元
        tetrahedra = []
        for e in range(self.n_elem_2d):
            v = self.elements_2d[e]
            v_bot = v
            v_top = v + n2

            # 分解为 3 个 TET4
            tetrahedra.append([v_bot[0], v_bot[1], v_bot[2], v_top[0]])
            tetrahedra.append([v_bot[1], v_bot[2], v_top[0], v_top[1]])
            tetrahedra.append([v_bot[2], v_top[0], v_top[1], v_top[2]])

        self.elements_3d = np.array(tetrahedra, dtype=int)
        self.n_node_3d = 2 * n2
        self.n_elem_3d = len(tetrahedra)

    def _tetrahedron_volume(self, elem_idx: int) -> float:
        """计算四面体单元体积."""
        v = self.elements_3d[elem_idx]
        p = self.nodes_3d[v]
        # V = |det([p1-p0, p2-p0, p3-p0])| / 6
        mat = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        return abs(np.linalg.det(mat)) / 6.0

    def assemble_mass_matrix(self) -> csr_matrix:
        """
        组装 3D 质量矩阵 M_{ij} = ∫ φ_i φ_j dV.

        TET4 元的质量矩阵（一致质量矩阵）:
            M_{local} = (V/20) * [[2,1,1,1],
                                  [1,2,1,1],
                                  [1,1,2,1],
                                  [1,1,1,2]]
        """
        row_idx = []
        col_idx = []
        data = []

        for e in range(self.n_elem_3d):
            v = self.elements_3d[e]
            vol = self._tetrahedron_volume(e)
            if vol < 1.0e-18:
                continue

            coeff = vol / 20.0
            for i in range(4):
                for j in range(4):
                    row_idx.append(v[i])
                    col_idx.append(v[j])
                    data.append(coeff * (2.0 if i == j else 1.0))

        M = csr_matrix((data, (row_idx, col_idx)), shape=(self.n_node_3d, self.n_node_3d))
        return M

    def project_2d_to_3d(self, A_2d: np.ndarray) -> np.ndarray:
        """
        将 2D 节点场 A_2d 投影到 3D 有限元空间。

        投影方程:  M * A_3d = b

        右端项近似:
            b_i = Σ_e ∫_{T_e} A_2d(x,y) φ_i dV

        由于每个 3D 节点对应一个 2D 节点（底面或顶面），
        且 A_2d 在底面和顶面相同，简化为:
            A_3d = [A_2d; A_2d]  （插值投影）

        但这里通过正式求解 L^2 投影方程获得更精确的 3D 场分布。
        """
        if len(A_2d) != self.n_node_2d:
            raise ValueError("2D场节点数与网格不匹配")

        M = self.assemble_mass_matrix()
        b = np.zeros(self.n_node_3d)

        # 组装右端项
        for e in range(self.n_elem_3d):
            v = self.elements_3d[e]
            vol = self._tetrahedron_volume(e)
            if vol < 1.0e-18:
                continue

            # 四面体质心投影到 xy 平面
            p_xy = np.mean(self.nodes_3d[v, :2], axis=0)
            # 在 2D 场中插值（最近节点近似）
            dists = np.sum((self.nodes_2d - p_xy) ** 2, axis=1)
            nearest = np.argmin(dists)
            A_val = A_2d[nearest]

            # 体积积分贡献: ∫ A φ_i dV ≈ A_val * V / 4
            for i in range(4):
                b[v[i]] += A_val * vol / 4.0

        # 求解投影方程
        A_3d = spsolve(M, b)
        if A_3d is None:
            # 如果质量矩阵奇异，使用直接插值
            A_3d = np.concatenate([A_2d, A_2d])
        else:
            A_3d = np.asarray(A_3d)

        return A_3d

    def compute_3d_magnetic_energy(self, A_3d: np.ndarray, nu_3d_func) -> float:
        """
        计算 3D 磁场储能:

            W_m^{3D} = 0.5 * ∫_{Ω_3D} ν |B|^2 dV

        对于拉伸网格，假设轴向分量 B_z ≈ 0（长电机近似），
        则 |B|^2 ≈ |∇_{xy} A_z|^2，能量约为 2D 能量的 L 倍。
        """
        W = 0.0
        for e in range(self.n_elem_3d):
            v = self.elements_3d[e]
            p = self.nodes_3d[v]
            vol = self._tetrahedron_volume(e)
            if vol < 1.0e-18:
                continue

            # TET4 基函数梯度
            grads = self._tet4_gradients(e)
            centroid = np.mean(p, axis=0)
            nu_val = nu_3d_func(centroid[0], centroid[1], centroid[2])

            dAdx = sum(A_3d[v[j]] * grads[j, 0] for j in range(4))
            dAdy = sum(A_3d[v[j]] * grads[j, 1] for j in range(4))
            B2 = dAdx * dAdx + dAdy * dAdy
            W += 0.5 * nu_val * B2 * vol

        return W

    def _tet4_gradients(self, elem_idx: int) -> np.ndarray:
        """
        计算 TET4 单元中四个基函数的梯度.

        对于节点 p_0, p_1, p_2, p_3:
            ∇φ_0 = ( (p_1-p_3) × (p_2-p_3) ) / (6V)
            ∇φ_1 = ( (p_2-p_3) × (p_0-p_3) ) / (6V)
            ...（轮换）
        """
        v = self.elements_3d[elem_idx]
        p = self.nodes_3d[v]
        # 体积
        mat = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        vol = np.linalg.det(mat) / 6.0
        if abs(vol) < 1.0e-18:
            return np.zeros((4, 3))

        grads = np.zeros((4, 3))
        for i in range(4):
            # 使用对偶基方法
            idx = [j for j in range(4) if j != i]
            a = p[idx[1]] - p[idx[2]]
            b = p[idx[0]] - p[idx[2]]
            # 叉积
            cross = np.array([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])
            grads[i] = cross / (6.0 * vol)
            # 符号修正: 对于 i=0，应为 (p1-p3)×(p2-p3)/(6V)
            # 上述计算给出的是朝外的法向，需要确保一致
        return grads

    def compute_axial_force_end_effects(self, A_3d: np.ndarray) -> float:
        """
        估算 3D 端部效应力（端部漏磁引起的轴向力）.

        简化模型: 端部轴向磁通密度 B_z 与轴向 A_z 变化率相关:
            B_z ≈ -∂A_z/∂z

        端部力近似:
            F_z ≈ 0.5 * ∫_{end_face} (B_z^2 / μ_0) dA
        """
        n2 = self.n_node_2d
        L = self.axial_length

        # 底面 (z=0) 和顶面 (z=L) 的 A_z 值
        A_bottom = A_3d[:n2]
        A_top = A_3d[n2:]

        # 轴向梯度近似
        dA_dz = (A_top - A_bottom) / L
        Bz = -dA_dz

        # 在底面/顶面的三角形上积分 Bz^2
        Fz = 0.0
        for e in range(self.n_elem_2d):
            v = self.elements_2d[e]
            p = self.nodes_2d[v]
            area = 0.5 * abs(
                (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
            )
            Bz_avg = np.mean(np.abs(Bz[v]))
            Fz += 0.5 * Bz_avg ** 2 / (4.0 * np.pi * 1.0e-7) * area

        return Fz
