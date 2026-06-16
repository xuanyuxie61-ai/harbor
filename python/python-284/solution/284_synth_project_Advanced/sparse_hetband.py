# -*- coding: utf-8 -*-
"""
sparse_hetband.py — 稀疏哈密顿量组装
=======================================
核心科学问题: 将高阶有限差分格式离散化的薛定谔方程
组装为稀疏矩阵特征值问题 H·ψ = E·ψ.

融合种子项目:
  - 992_r8ri: R8RI 稀疏矩阵存储格式

稀疏矩阵格式:
  使用 CSR (Compressed Sparse Row) 格式存储带状哈密顿量.
  对于 2p 阶差分, 带宽 = 2p+1.

  对于 N 个网格点, 矩阵有 O(pN) 个非零元素.
"""

import numpy as np
from high_order_fd import HBAR, M0, E0, BenDanielDukeFD, D2_STENCILS


class SparseBandMatrix:
    """
    稀疏带状矩阵 (CSR 格式).
    融合 992_r8ri: Row-Indexed 稀疏存储.

    存储:
      data[i] : 第 i 行非零元素值
      col_idx[i] : 第 i 行非零元素列索引
      row_ptr[i] : 第 i 行在 data 中的起始位置
    """

    def __init__(self, N):
        self.N = N
        self.data = []
        self.col_idx = []
        self.row_ptr = [0]

    def add_element(self, row, col, value):
        """添加矩阵元素 (必须在组装阶段调用)."""
        if abs(value) < 1e-30:
            return
        self.data.append(value)
        self.col_idx.append(col)

    def finalize(self, row_indices):
        """
        完成矩阵组装.

        参数
        ----
        row_indices : list of list
            每行的 (col, val) 对列表
        """
        self.data = []
        self.col_idx = []
        self.row_ptr = [0]

        for row_data in row_indices:
            # 按列排序
            row_data.sort(key=lambda x: x[0])
            for col, val in row_data:
                if abs(val) > 1e-30:
                    self.data.append(val)
                    self.col_idx.append(col)
            self.row_ptr.append(len(self.data))

        self.data = np.array(self.data)
        self.col_idx = np.array(self.col_idx, dtype=np.int32)
        self.row_ptr = np.array(self.row_ptr, dtype=np.int32)

    def matvec(self, x):
        """稀疏矩阵-向量乘法 y = A·x."""
        N = self.N
        y = np.zeros(N)
        for i in range(N):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for j in range(start, end):
                y[i] += self.data[j] * x[self.col_idx[j]]
        return y

    def get_diagonal(self):
        """提取对角线元素."""
        diag = np.zeros(self.N)
        for i in range(self.N):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for j in range(start, end):
                if self.col_idx[j] == i:
                    diag[i] = self.data[j]
        return diag

    def nnz(self):
        """非零元素数."""
        return len(self.data)

    def to_dense(self):
        """转换为稠密矩阵 (仅用于小规模问题)."""
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for j in range(start, end):
                A[i, self.col_idx[j]] = self.data[j]
        return A


class HeteroHamiltonianAssembler:
    """
    异质结哈密顿量组装器.

    H = T + V
    T = -(ℏ²/2) d/dz [1/m*(z) d/dz]  (Ben Daniel-Duke 动能)
    V = V_conduction(z) 或 V_valence(z)  (异质结势能)
    """

    def __init__(self, z_grid, m_star_profile, V_profile, order=4):
        """
        参数
        ----
        z_grid : ndarray
            空间网格 [m]
        m_star_profile : ndarray
            有效质量剖面 [m0]
        V_profile : ndarray
            势能剖面 [eV]
        order : int
            差分精度
        """
        self.z = z_grid
        self.m_star = m_star_profile
        self.V = V_profile
        self.N = len(z_grid)
        self.dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 1e-10
        self.fd = BenDanielDukeFD(order)
        self.order = order

    def assemble(self):
        """
        组装哈密顿量矩阵 H [eV].

        H_ij = T_ij + V_i * delta_ij

        T_ij 由 Ben Daniel-Duke 格式给出:
          对角: T_ii = (ℏ²/(2m0·dz²)) * (1/m*_{i+1/2} + 1/m*_{i-1/2})
          上对角: T_{i,i+1} = -(ℏ²/(2m0·dz²)) * 1/m*_{i+1/2}
          下对角: T_{i-1,i} = -(ℏ²/(2m0·dz²)) * 1/m*_{i-1/2}

        返回
        ----
        SparseBandMatrix
        """
        N = self.N
        dz = self.dz

        # 半格点 1/m*
        inv_m = np.zeros(N + 1)
        for i in range(N):
            inv_m[i] = 1.0 / max(self.m_star[i], 1e-8)
        inv_m[N] = inv_m[N - 1]
        # 调和平均到半格点
        inv_m_half = np.zeros(N + 1)
        for i in range(N):
            inv_m_half[i] = 0.5 * (inv_m[i] + inv_m[i + 1])
        inv_m_half[0] = inv_m[0]
        inv_m_half[N] = inv_m[N - 1]

        # 动能系数 [eV]
        # T_0 = ℏ²/(2·m0·dz²) [J] → [eV]
        T0 = HBAR**2 / (2.0 * M0 * dz**2 * E0)

        # 组装行数据
        row_data = [[] for _ in range(N)]

        # 内部点
        for i in range(1, N - 1):
            # 对角: T_ii + V_i
            T_diag = T0 * (inv_m_half[i] + inv_m_half[i - 1])
            row_data[i].append((i, T_diag + self.V[i]))

            # 上对角
            T_upper = -T0 * inv_m_half[i]
            row_data[i].append((i + 1, T_upper))

            # 下对角
            T_lower = -T0 * inv_m_half[i - 1]
            row_data[i].append((i - 1, T_lower))

        # 边界点 (Dirichlet: ψ=0)
        # H[0,0] = T0/m_half[0] + V[0], H[0,1] = -T0/m_half[0]
        row_data[0].append((0, T0 * inv_m_half[0] + self.V[0]))
        row_data[0].append((1, -T0 * inv_m_half[0]))
        row_data[-1].append((-2, -T0 * inv_m_half[-2]))
        row_data[-1].append((-1, T0 * inv_m_half[-1] + self.V[-1]))

        # 高阶修正 (4阶)
        if self.order >= 4 and N >= 5:
            row_data = self._add_high_order_correction(
                row_data, T0, inv_m_half)

        # 构建稀疏矩阵
        H = SparseBandMatrix(N)
        H.finalize(row_data)
        return H

    def _add_high_order_correction(self, row_data, T0, inv_m_half):
        """
        4阶差分修正:
        额外贡献来自 5 点模板的非最近邻元素.

        4阶 BDD 格式额外项:
          T_{i,i+2} += T0/(12) * 1/m*_{i+3/2}
          T_{i,i-2} += T0/(12) * 1/m*_{i-3/2}
          T_{i,i+1} += T0*(1/12) * (1/m*_{i+1/2}) 修正
        """
        N = self.N
        correction = T0 / 12.0

        for i in range(2, N - 2):
            # i+2 元素
            if i + 2 < N:
                row_data[i].append((i + 2, correction * inv_m_half[min(i + 1, N - 1)]))
            # i-2 元素
            if i - 2 >= 0:
                row_data[i].append((i - 2, correction * inv_m_half[max(i - 1, 0)]))

            # 修正 i+1, i-1
            for idx, (col, val) in enumerate(row_data[i]):
                if col == i + 1:
                    row_data[i][idx] = (col, val - correction * inv_m_half[i])
                elif col == i - 1:
                    row_data[i][idx] = (col, val - correction * inv_m_half[i - 1])

        return row_data

    def assemble_potential_matrix(self):
        """组装势能对角矩阵 V."""
        N = self.N
        row_data = [[(i, self.V[i])] for i in range(N)]
        V_mat = SparseBandMatrix(N)
        V_mat.finalize(row_data)
        return V_mat


class MultiBandHamiltonian:
    """
    多带哈密顿量 (耦合导带-价带).

    用于窄带隙半导体, 需要考虑导带-价带耦合:
        H = [H_cc   H_cv]
            [H_vc   H_vv]

    H_cc: 导带块 (N×N)
    H_vv: 价带块 (N×N)
    H_cv: 耦合块 (N×N), 正比于 ℏ·P/m0

    Kane 参数 P:
        P = -iℏ/m0 <S|p_x|X> ≈ ℏ/m0 * sqrt(m0*Ep/2)
        Ep ≈ 20 eV (典型 TMDC 值)
    """

    def __init__(self, z_grid, m_e_profile, m_h_profile,
                 Vc_profile, Vv_profile, Ep=20.0):
        self.z = z_grid
        self.N = len(z_grid)
        self.m_e = m_e_profile
        self.m_h = m_h_profile
        self.Vc = Vc_profile
        self.Vv = Vv_profile
        self.Ep = Ep  # Kane 能量 [eV]
        self.dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 1e-10

    def kane_momentum(self):
        """
        Kane 动量矩阵元:
            P = sqrt(m0 * Ep * E0 / 2) [kg·m/s]
        其中 Ep 是 Kane 能量, E0 是元电荷.
        """
        return np.sqrt(M0 * self.Ep * E0 / 2.0)

    def coupling_strength(self):
        """
        导带-价带耦合强度:
            ℏP/m0 = ℏ/m0 * sqrt(m0*Ep*e/2)
                  = sqrt(ℏ²*Ep*e/(2*m0))
        """
        P = self.kane_momentum()
        return HBAR * P / M0

    def assemble_2band(self):
        """
        组装 2N×2N 多带哈密顿量.
        返回稠密矩阵 (仅用于小规模验证).
        """
        # 导带块
        Hc_asm = HeteroHamiltonianAssembler(
            self.z, self.m_e, self.Vc, order=2)
        Hc = Hc_asm.assemble().to_dense()

        # 价带块
        Hv_asm = HeteroHamiltonianAssembler(
            self.z, self.m_h, self.Vv, order=2)
        Hv = Hv_asm.assemble().to_dense()

        N = self.N
        H_full = np.zeros((2 * N, 2 * N))
        H_full[:N, :N] = Hc
        H_full[N:, N:] = Hv

        # k·p 耦合 (一阶差分近似 p_z)
        P_coupling = self.coupling_strength() / E0  # [eV·m]
        for i in range(N - 1):
            coupling = P_coupling / (2 * self.dz)
            H_full[i, N + i + 1] += coupling
            H_full[i, N + i - 1] += -coupling if i > 0 else 0
            H_full[N + i + 1, i] += coupling
            if i > 0:
                H_full[N + i - 1, i] += -coupling

        return H_full
