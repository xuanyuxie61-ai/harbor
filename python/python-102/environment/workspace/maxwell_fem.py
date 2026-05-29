"""
maxwell_fem.py
==============
基于有限元方法（FEM）求解二维横磁（TM）模式下超构表面纳米柱单元的
电磁散射问题，即亥姆霍兹方程（Helmholtz equation）。

本模块源自项目 408_fem2d_poisson_rectangle 的核心思想，
将泊松方程 -∇²u = f 推广至复值亥姆霍兹方程：
    -∇²E_z - k₀² ε_r(x,y) E_z = f_src
其中 k₀ = 2π/λ₀ 为自由空间波数，ε_r 为相对介电常数分布。

采用六节点二次三角形单元（T6），使用稀疏矩阵存储，
支持完美匹配层（PML）截断边界条件。
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class MaxwellFEM2D:
    """
    2D 横磁模式（TM, E_z 极化）亥姆霍兹方程有限元求解器。

    控制方程（频域麦克斯韦方程组简化）：
        ∇·(μ_r⁻¹ ∇E_z) + k₀² ε_r E_z = -j ω μ₀ J_z
    对于非磁性介质（μ_r = 1），简化为：
        ∇²E_z + k₀² ε_r E_z = f_src

    弱形式：
        ∫_Ω (∇w·∇E_z - k₀² ε_r w E_z) dΩ
        + ∫_Γ_PML σ(x,y) w E_z dΓ = -∫_Ω w f_src dΩ
    """

    def __init__(self, wavelength=1.55e-6, n_si=3.48, n_air=1.0,
                 pml_width=0.5e-6, pml_sigma_max=1.0e7):
        """
        Parameters
        ----------
        wavelength : float
            自由空间波长 λ₀ [m]，默认 1.55 μm（光通信C波段）
        n_si : float
            硅（Si）在 λ₀ 处的折射率
        n_air : float
            空气（或基底）折射率
        pml_width : float
            PML 层厚度 [m]
        pml_sigma_max : float
            PML 最大电导率 [S/m]
        """
        self.wavelength = wavelength
        self.k0 = 2.0 * np.pi / wavelength  # 自由空间波数
        self.n_si = n_si
        self.n_air = n_air
        self.eps_si = n_si ** 2
        self.eps_air = n_air ** 2
        self.pml_width = pml_width
        self.pml_sigma_max = pml_sigma_max
        self.c = 2.99792458e8  # 光速 [m/s]
        self.omega = 2.0 * np.pi * self.c / wavelength

    # ------------------------------------------------------------------
    # 几何与网格生成
    # ------------------------------------------------------------------
    def build_rectangular_mesh(self, nx, ny, xlim, ylim):
        """
        在矩形区域 [xlim]×[ylim] 上构建 T6（六节点二次三角形）网格。

        Returns
        -------
        nodes : ndarray, shape (node_num, 2)
        elements : ndarray, shape (element_num, 6), 0-based
        """
        xl, xr = xlim
        yb, yt = ylim
        dx = (xr - xl) / (nx - 1)
        dy = (yt - yb) / (ny - 1)

        # 角节点 + 边中节点
        node_num = (2 * nx - 1) * (2 * ny - 1)
        nodes = np.zeros((node_num, 2), dtype=np.float64)

        idx = 0
        for j in range(2 * ny - 1):
            for i in range(2 * nx - 1):
                if j % 2 == 0:
                    y = yb + (j // 2) * dy
                else:
                    y = yb + (j // 2) * dy + dy / 2.0
                if i % 2 == 0:
                    x = xl + (i // 2) * dx
                else:
                    x = xl + (i // 2) * dx + dx / 2.0
                nodes[idx] = [x, y]
                idx += 1

        element_num = 2 * (nx - 1) * (ny - 1)
        elements = np.zeros((element_num, 6), dtype=np.int32)
        e = 0
        for j in range(ny - 1):
            for i in range(nx - 1):
                sw = j * 2 * (2 * nx - 1) + 2 * i
                w = sw + 1
                nw = sw + 2
                s = sw + (2 * nx - 1)
                c = s + 1
                n = s + 2
                se = s + (2 * nx - 1)
                ee = se + 1
                ne = se + 2
                # 两个三角形单元，节点编号从 1 转为 0
                elements[e] = [sw, se, nw, s, c, w]
                e += 1
                elements[e] = [ne, nw, se, n, c, ee]
                e += 1

        return nodes, elements

    # ------------------------------------------------------------------
    # 介质分布与 PML
    # ------------------------------------------------------------------
    def epsilon_profile(self, x, y, pillar_center, pillar_size):
        """
        定义矩形纳米柱的相对介电常数分布 ε_r(x,y)。

        纳米柱区域：|x - cx| ≤ w/2 且 |y - cy| ≤ h/2 时 ε_r = n_si²，
        否则 ε_r = n_air²。
        """
        # TODO: 根据超构表面纳米柱几何模型实现截面形状判断与介电常数赋值
        # 提示：pillar_size 的维度与形状定义需要与 phase_quadrature.py 中的定义保持一致
        cx, cy = pillar_center
        w, h = pillar_size
        ...
        eps = ...
        return eps.astype(np.complex128)

    def pml_stretch(self, x, y, xlim, ylim):
        """
        计算 PML 复坐标拉伸因子 s(x) = 1 + σ(x)/(jω)。
        采用多项式吸收轮廓：σ(ρ) = σ_max * (ρ/d)^m，m=2。

        Returns
        -------
        sx, sy : complex
        """
        xl, xr = xlim
        yb, yt = ylim
        m_order = 2
        sx = 1.0 + 0.0j
        sy = 1.0 + 0.0j

        if x < xl + self.pml_width:
            d = xl + self.pml_width - x
            sigma = self.pml_sigma_max * (d / self.pml_width) ** m_order
            sx = 1.0 - 1.0j * sigma / self.omega
        elif x > xr - self.pml_width:
            d = x - (xr - self.pml_width)
            sigma = self.pml_sigma_max * (d / self.pml_width) ** m_order
            sx = 1.0 - 1.0j * sigma / self.omega

        if y < yb + self.pml_width:
            d = yb + self.pml_width - y
            sigma = self.pml_sigma_max * (d / self.pml_width) ** m_order
            sy = 1.0 - 1.0j * sigma / self.omega
        elif y > yt - self.pml_width:
            d = y - (yt - self.pml_width)
            sigma = self.pml_sigma_max * (d / self.pml_width) ** m_order
            sy = 1.0 - 1.0j * sigma / self.omega

        return sx, sy

    # ------------------------------------------------------------------
    # 基函数与数值积分
    # ------------------------------------------------------------------
    @staticmethod
    def quad_points_t6():
        """
        三角形上的 7 点高斯积分规则（精度 5 阶）。
        参考：Dunavant 1985。
        """
        # 面积坐标 (L1, L2, L3)，其中 L3 = 1 - L1 - L2
        qp = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
        ], dtype=np.float64)
        wq = np.array([
            0.225000000000000,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ], dtype=np.float64)
        return qp, wq

    def shape_t6(self, r, s):
        """
        参考单元上的二次三角形基函数及其梯度（对参考坐标）。
        参考单元顶点：(0,0), (1,0), (0,1)。

        Returns
        -------
        N : ndarray, shape (6,)
        dNdr : ndarray, shape (6,)
        dNds : ndarray, shape (6,)
        """
        t = 1.0 - r - s
        N = np.array([
            2.0 * t * (t - 0.5),
            2.0 * r * (r - 0.5),
            2.0 * s * (s - 0.5),
            4.0 * r * t,
            4.0 * r * s,
            4.0 * s * t,
        ], dtype=np.float64)
        dNdr = np.array([
            -3.0 + 4.0 * r + 4.0 * s,
            4.0 * r - 1.0,
            0.0,
            4.0 * (1.0 - 2.0 * r - s),
            4.0 * s,
            -4.0 * s,
        ], dtype=np.float64)
        dNds = np.array([
            -3.0 + 4.0 * r + 4.0 * s,
            0.0,
            4.0 * s - 1.0,
            -4.0 * r,
            4.0 * r,
            4.0 * (1.0 - r - 2.0 * s),
        ], dtype=np.float64)
        return N, dNdr, dNds

    # ------------------------------------------------------------------
    # 矩阵组装
    # ------------------------------------------------------------------
    def assemble_system(self, nodes, elements, pillar_center, pillar_size,
                        xlim, ylim, source_type='plane_wave'):
        """
        组装全局刚度矩阵 A 和右端项 b。

        离散弱形式（PML 拉伸后）：
            ∫ (sy/sx * ∂w/∂x * ∂E/∂x + sx/sy * ∂w/∂y * ∂E/∂y) dΩ
            - k₀² ∫ sx*sy * ε_r * w * E dΩ = -∫ w * f_src dΩ

        Returns
        -------
        A : scipy.sparse.csc_matrix
        b : ndarray, shape (node_num,)
        """
        node_num = nodes.shape[0]
        element_num = elements.shape[0]
        qp, wq = self.quad_points_t6()

        row_idx = []
        col_idx = []
        data = []
        b = np.zeros(node_num, dtype=np.complex128)

        for e in range(element_num):
            en = elements[e]  # 6 local node indices
            xn = nodes[en, 0]
            yn = nodes[en, 1]

            # 计算单元面积（雅可比行列式的一半）
            # 参考单元到物理单元的映射：
            # x = Σ N_i(r,s) * x_i, y = Σ N_i(r,s) * y_i
            # J = [[dx/dr, dx/ds], [dy/dr, dy/ds]]
            x1, x2, x3 = xn[0], xn[1], xn[2]
            y1, y2, y3 = yn[0], yn[1], yn[2]
            detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if abs(detJ) < 1e-18:
                continue
            area = 0.5 * abs(detJ)

            # 常数雅可比近似（线性部分）用于二次映射的简化
            # 更精确地，在每个积分点重新计算
            for iq in range(len(wq)):
                L1, L2, L3 = qp[iq]
                r = L2
                s = L3
                w = area * wq[iq]

                N, dNdr, dNds = self.shape_t6(r, s)

                # 物理坐标
                x = np.dot(N, xn)
                y = np.dot(N, yn)

                # 雅可比矩阵
                dxdr = np.dot(dNdr, xn)
                dxds = np.dot(dNds, xn)
                dydr = np.dot(dNdr, yn)
                dyds = np.dot(dNds, yn)
                detJ_q = dxdr * dyds - dxds * dydr
                if abs(detJ_q) < 1e-18:
                    continue

                # 梯度变换：dN/dx = (dN/dr * dy/ds - dN/ds * dy/dr) / detJ
                inv_det = 1.0 / detJ_q
                dNdx = (dNdr * dyds - dNds * dydr) * inv_det
                dNdy = (dNds * dxdr - dNdr * dxds) * inv_det

                # 材料与 PML
                eps = self.epsilon_profile(x, y, pillar_center, pillar_size)
                sx, sy = self.pml_stretch(x, y, xlim, ylim)

                # 刚度矩阵贡献
                for i in range(6):
                    for j in range(6):
                        aij = (sy / sx) * dNdx[i] * dNdx[j] + (sx / sy) * dNdy[i] * dNdy[j]
                        aij -= self.k0 ** 2 * sx * sy * eps * N[i] * N[j]
                        aij *= w * abs(detJ_q)
                        row_idx.append(en[i])
                        col_idx.append(en[j])
                        data.append(aij)

                # 右端项（入射平面波源）
                if source_type == 'plane_wave':
                    # E_inc = exp(j k0 x)
                    f_val = -self.k0 ** 2 * (eps - self.eps_air) * np.exp(1.0j * self.k0 * x)
                else:
                    f_val = 0.0
                for i in range(6):
                    b[en[i]] += w * abs(detJ_q) * f_val * N[i]

        A = sparse.coo_matrix((data, (row_idx, col_idx)),
                               shape=(node_num, node_num), dtype=np.complex128).tocsc()
        return A, b

    def apply_dirichlet_boundary(self, A, b, nodes, xlim, ylim):
        """
        在矩形外边界上施加一阶吸收边界条件（ABC）近似：
            ∂E/∂n + j k0 E ≈ 0
        简化处理：直接在边界节点上设 E = 0（完美电导体近似，
        用于内部问题；外部散射问题使用 PML，此处仅做边界固定）。
        """
        xl, xr = xlim
        yb, yt = ylim
        tol = 1e-12
        boundary_nodes = []
        for i, (x, y) in enumerate(nodes):
            if (abs(x - xl) < tol or abs(x - xr) < tol or
                    abs(y - yb) < tol or abs(y - yt) < tol):
                boundary_nodes.append(i)

        boundary_nodes = np.array(boundary_nodes, dtype=np.int32)
        if len(boundary_nodes) == 0:
            return A, b

        # 强制边界值为 0
        for idx in boundary_nodes:
            # 清零整行和整列，对角线置 1
            # 使用 lil 格式更方便行操作
            pass

        A_lil = A.tolil()
        for idx in boundary_nodes:
            A_lil.rows[idx] = [idx]
            A_lil.data[idx] = [1.0 + 0.0j]
            b[idx] = 0.0 + 0.0j

        # 清理列（保持对称性近似）
        for i in range(A_lil.shape[0]):
            if i in boundary_nodes:
                continue
            new_rows = []
            new_data = []
            for j, val in zip(A_lil.rows[i], A_lil.data[i]):
                if j not in boundary_nodes:
                    new_rows.append(j)
                    new_data.append(val)
            A_lil.rows[i] = new_rows
            A_lil.data[i] = new_data

        return A_lil.tocsc(), b

    def solve_scattering(self, nx=25, ny=25,
                         domain=(-2.0e-6, 2.0e-6, -2.0e-6, 2.0e-6),
                         pillar_center=(0.0, 0.0),
                         pillar_size=(0.5e-6, 1.0e-6)):
        """
        完整求解流程：网格生成 → 矩阵组装 → 施加边界 → 求解。

        Returns
        -------
        E_z : ndarray, shape (node_num,), 复数场分布
        nodes : ndarray, mesh nodes
        elements : ndarray, mesh elements
        """
        xlim = (domain[0], domain[1])
        ylim = (domain[2], domain[3])
        nodes, elements = self.build_rectangular_mesh(nx, ny, xlim, ylim)
        A, b = self.assemble_system(nodes, elements, pillar_center, pillar_size,
                                     xlim, ylim)
        A, b = self.apply_dirichlet_boundary(A, b, nodes, xlim, ylim)
        E_z = spsolve(A, b)
        return E_z, nodes, elements

    def compute_transmission_phase(self, E_z, nodes, y_eval):
        """
        在 y = y_eval 的直线上提取透射场相位。
        使用节点线性插值。
        """
        # 找到 y 坐标接近 y_eval 的节点
        tol = 1e-12
        line_nodes = np.where(np.abs(nodes[:, 1] - y_eval) < tol)[0]
        if len(line_nodes) == 0:
            # 最近邻插值
            line_nodes = np.argsort(np.abs(nodes[:, 1] - y_eval))[:nodes.shape[0] // nx]
        x_vals = nodes[line_nodes, 0]
        E_vals = E_z[line_nodes]
        # 按 x 排序
        order = np.argsort(x_vals)
        return x_vals[order], E_vals[order]


def demo():
    """简单演示：求解单个纳米柱的散射场。"""
    fem = MaxwellFEM2D(wavelength=1.55e-6)
    E_z, nodes, elements = fem.solve_scattering(
        nx=17, ny=17,
        domain=(-1.5e-6, 1.5e-6, -1.5e-6, 1.5e-6),
        pillar_center=(0.0, 0.0),
        pillar_size=(0.3e-6, 0.6e-6)
    )
    phase = np.angle(E_z)
    print(f"[maxwell_fem] 节点数: {len(E_z)}, 相位范围: [{phase.min():.4f}, {phase.max():.4f}] rad")
    # 计算平均透射相位（中心区域）
    cx, cy = 0.0, 0.0
    dist = np.sqrt((nodes[:, 0] - cx) ** 2 + (nodes[:, 1] - cy) ** 2)
    mask = dist < 0.2e-6
    if mask.any():
        avg_phase = np.angle(np.mean(E_z[mask]))
        print(f"[maxwell_fem] 中心区域平均透射相位: {avg_phase:.4f} rad = {np.degrees(avg_phase):.2f}°")
    return E_z, nodes, elements


if __name__ == "__main__":
    demo()
