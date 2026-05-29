"""
fem1d_radial.py
================================================================================
1D 径向有限元求解器：电机气隙磁势分布的一维简化模型

融合原项目:
  - 383_fem1d : 1D有限元方法（线性基函数、刚度矩阵组装、三对角求解、边界条件处理）

核心科学内容:
  1. 一维径向静磁场方程（柱坐标下的轴对称简化）:

        - d/dr ( ν(r) * r * dA_θ/dr ) + (ν(r)/r) * A_θ = J_s(r)

     其中 A_θ 为周向磁矢势，ν = 1/μ 为磁阻率。

  2. 分段线性基函数（P1 Lagrange元）:

        φ_i(r) = (r_{i+1} - r) / h_i,   r ∈ [r_i, r_{i+1}]
        φ_{i+1}(r) = (r - r_i) / h_i,   r ∈ [r_i, r_{i+1}]

  3. 单元刚度矩阵（局部坐标 ξ ∈ [0,1], r = r_i + h_i ξ）:

        K_{ij}^{(e)} = ∫_{r_i}^{r_{i+1}} [ ν(r) r dφ_i/dr dφ_j/dr + (ν(r)/r) φ_i φ_j ] dr

  4. 边界条件:
        - Dirichlet: A_θ(R_outer) = 0  （定子外圆磁势参考零点）
        - Neumann:   dA_θ/dr|_{R_inner} = 0  （轴心对称条件）

  5.  Thomas 算法求解三对角线性系统（O(n)复杂度）.
================================================================================
"""

import numpy as np


class FEM1DRadial:
    """
    1D 径向有限元求解器，用于电机气隙区域的一维磁势分析。

    融合原项目 383_fem1d 的核心算法:
      - init / geometry / assemble / solve / output 流程
      - phi 线性基函数及其导数
      - 三对角矩阵 Thomas 求解
    """

    def __init__(self, r_min: float, r_max: float, n_elements: int):
        if r_min <= 0.0:
            raise ValueError("径向内半径必须为正")
        if r_max <= r_min:
            raise ValueError("径向外半径必须大于内半径")
        if n_elements < 2:
            raise ValueError("单元数至少为2")
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.n_elements = int(n_elements)
        self.n_nodes = n_elements + 1

        # 均匀网格节点
        self.nodes = np.linspace(r_min, r_max, self.n_nodes)
        self.h = np.diff(self.nodes)

        # 边界条件标记
        # ibc=1: 左端 Dirichlet, 右端 Neumann
        self.ibc = 1
        self.ul = 0.0  # A_θ(r_max) = 0
        self.ur = 0.0  # dA_θ/dr(r_min) = 0

        # 未知数映射: indx[i] = 节点i对应的未知数编号，-1 表示 Dirichlet 固定
        self.indx = np.zeros(self.n_nodes, dtype=int)
        self._setup_indices()

    def _setup_indices(self):
        """建立节点到未知数的映射，处理边界条件."""
        nu = 0
        # 左端点 (r_min): Neumann → 有未知数
        self.indx[0] = nu
        nu += 1
        # 内部节点
        for i in range(1, self.n_elements):
            self.indx[i] = nu
            nu += 1
        # 右端点 (r_max): Dirichlet → 固定
        self.indx[self.n_elements] = -1
        self.nu = nu

    def _basis(self, ie: int, il: int, r: float) -> tuple:
        """
        评估单元 ie 中的第 il 个线性基函数及其导数.

        il = 0: 左节点基函数
        il = 1: 右节点基函数
        """
        r_left = self.nodes[ie]
        r_right = self.nodes[ie + 1]
        h_e = self.h[ie]
        if r < r_left or r > r_right:
            return 0.0, 0.0
        if il == 0:
            phi = (r_right - r) / h_e
            dphi = -1.0 / h_e
        else:
            phi = (r - r_left) / h_e
            dphi = 1.0 / h_e
        return phi, dphi

    def assemble(
        self, nu_func, source_func, nquad: int = 3
    ) -> tuple:
        """
        组装刚度矩阵和载荷向量.

        参数:
            nu_func    : callable(r) -> float, 磁阻率 ν(r) = 1/μ(r)
            source_func: callable(r) -> float, 源电流密度 J_s(r) [A/m^2]
            nquad      : 每单元高斯积分点数

        返回:
            adiag, aleft, arite, f  (三对角格式)
        """
        # 初始化
        f = np.zeros(self.nu)
        adiag = np.zeros(self.nu)
        aleft = np.zeros(self.nu)
        arite = np.zeros(self.nu)

        # 高斯积分点和权重（标准区间 [-1,1] 映射到 [r_left, r_right]）
        from quadrature_engine import GaussLegendreQuadrature

        glq = GaussLegendreQuadrature(nquad)

        for ie in range(self.n_elements):
            r_left = self.nodes[ie]
            r_right = self.nodes[ie + 1]
            h_e = self.h[ie]
            scale = 0.5 * h_e
            shift = 0.5 * (r_left + r_right)

            quad_pts = shift + scale * glq.nodes
            quad_wts = scale * glq.weights

            for iq in range(nquad):
                rq = quad_pts[iq]
                wq = quad_wts[iq]
                nu_r = nu_func(rq)
                js_r = source_func(rq)

                for il in range(2):
                    ig = ie + il
                    iu = self.indx[ig]
                    if iu < 0:
                        continue

                    phi_i, dphi_i = self._basis(ie, il, rq)
                    f[iu] += wq * js_r * phi_i

                    # 边界条件贡献（Neumann在左端自然满足，无需显式处理）

                    for jl in range(2):
                        jg = ie + jl
                        ju = self.indx[jg]
                        phi_j, dphi_j = self._basis(ie, jl, rq)

                        # 刚度矩阵项:
                        # K_ij = ∫ [ ν r dφ_i/dr dφ_j/dr + (ν/r) φ_i φ_j ] dr
                        aij = wq * (
                            nu_r * rq * dphi_i * dphi_j
                            + (nu_r / rq) * phi_i * phi_j
                        )

                        if ju < 0:
                            # Dirichlet 边界贡献移到右端
                            if jg == self.n_elements:
                                f[iu] -= aij * self.ul
                        elif iu == ju:
                            adiag[iu] += aij
                        elif ju < iu:
                            aleft[iu] += aij
                        else:
                            arite[iu] += aij

        return adiag, aleft, arite, f

    @staticmethod
    def solve_tridiagonal(adiag: np.ndarray, aleft: np.ndarray, arite: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Thomas 算法求解三对角系统 A u = f.

        A 的结构:
            第 i 行: aleft[i] * u_{i-1} + adiag[i] * u_i + arite[i] * u_{i+1} = f[i]

        前向消元:
            adiag'[i] = adiag[i] - aleft[i] * arite[i-1] / adiag'[i-1]
            f'[i] = f[i] - aleft[i] * f'[i-1] / adiag'[i-1]

        回代:
            u[n-1] = f'[n-1] / adiag'[n-1]
            u[i] = (f'[i] - arite[i] * u[i+1]) / adiag'[i]
        """
        n = len(f)
        adiag = adiag.copy()
        f = f.copy()

        # 边界保护
        if abs(adiag[0]) < 1.0e-30:
            adiag[0] = 1.0e-30 * np.sign(adiag[0] + 1.0e-30)

        # 前向消元
        for i in range(1, n):
            m = aleft[i] / adiag[i - 1]
            adiag[i] -= m * arite[i - 1]
            f[i] -= m * f[i - 1]
            if abs(adiag[i]) < 1.0e-30:
                adiag[i] = 1.0e-30 * np.sign(adiag[i] + 1.0e-30)

        # 回代
        u = np.zeros(n)
        u[-1] = f[-1] / adiag[-1]
        for i in range(n - 2, -1, -1):
            u[i] = (f[i] - arite[i] * u[i + 1]) / adiag[i]
        return u

    def solve(self, nu_func, source_func, nquad: int = 3) -> np.ndarray:
        """组装并求解完整系统，返回所有节点上的磁矢势值."""
        adiag, aleft, arite, f = self.assemble(nu_func, source_func, nquad)
        u_sol = self.solve_tridiagonal(adiag, aleft, arite, f)

        # 重构完整解向量（包含 Dirichlet 边界）
        A_full = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            idx = self.indx[i]
            if idx >= 0:
                A_full[i] = u_sol[idx]
            else:
                A_full[i] = self.ul
        return A_full

    def compute_radial_b_field(self, A: np.ndarray) -> np.ndarray:
        """
        由磁矢势 A_θ(r) 计算径向磁通密度 B_r = (1/r) d(r A_θ)/dr.

        在单元内，A_θ 线性插值:
            A_θ(r) = A_i φ_i(r) + A_{i+1} φ_{i+1}(r)
            dA_θ/dr = (A_{i+1} - A_i) / h_i

        因此:
            B_r(r) = A_θ(r) / r + dA_θ/dr
        """
        # TODO: 请实现径向磁通密度计算
        raise NotImplementedError("Hole_3: 需实现由 A_θ 计算 B_r 的公式")

    def compute_energy(self, A: np.ndarray, nu_func, nquad: int = 3) -> float:
        """
        计算磁场储能:

            W_m = 0.5 * ∫ ν(r) * B^2(r) * 2π r L dr

        其中 L 为轴向长度（取 L=1m 进行归一化）.
        """
        from quadrature_engine import GaussLegendreQuadrature

        glq = GaussLegendreQuadrature(nquad)
        W = 0.0
        L = 1.0  # 单位轴向长度

        for ie in range(self.n_elements):
            r_left = self.nodes[ie]
            r_right = self.nodes[ie + 1]
            h_e = self.h[ie]
            scale = 0.5 * h_e
            shift = 0.5 * (r_left + r_right)
            quad_pts = shift + scale * glq.nodes
            quad_wts = scale * glq.weights

            # 单元内 A_θ 线性
            A_i = A[ie]
            A_ip1 = A[ie + 1]
            dA = (A_ip1 - A_i) / h_e

            for iq in range(nquad):
                rq = quad_pts[iq]
                wq = quad_wts[iq]
                Aq = A_i + dA * (rq - r_left)
                # B_r = A_θ/r + dA_θ/dr
                Bq = Aq / rq + dA
                nu_r = nu_func(rq)
                W += 0.5 * nu_r * Bq * Bq * 2.0 * np.pi * rq * L * wq

        return W
