"""
sparse_acoustics.py
稀疏矩阵声学传播模型与高效矩阵运算

融合原始项目:
  - 1154_st_to_ccs (ST到CCS稀疏格式转换)
  - 1379_usa_matrix (稀疏邻接矩阵/图结构)
  - 1157_st_to_hb (Harwell-Boeing格式思想)

科学背景:
  在3D封闭空间中,声源到传感器的传播可用 room impulse response (RIR) 描述.
  对于N个次级声源和M个误差传感器,传播矩阵 H \in C^{M\times N} 通常是稀疏的
  (每个传感器仅与邻近声源有强耦合).

  本模块实现:
  1. 基于图论的稀疏声传播矩阵生成 (类似USA邻接图结构)
  2. ST (Sparse Triplet) 到 CCS (Compressed Column Storage) 转换
  3. 稀疏矩阵-向量乘法 (用于多通道ANC的前向传播)

  关键公式:
      声压向量:  p = H \cdot s
      其中 s 为次级声源信号向量, p 为传感器处声压.
"""

import numpy as np
from collections import defaultdict


class SparseAcousticMatrix:
    """
    稀疏声学传播矩阵,支持ST和CCS格式.
    """
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.st_rows = []
        self.st_cols = []
        self.st_vals = []
        self.ccs_ready = False
        self.ccc = None
        self.icc = None
        self.acc = None

    def add_entry(self, i, j, val):
        """添加ST格式元素."""
        if not (0 <= i < self.m and 0 <= j < self.n):
            raise IndexError("Sparse index out of bounds")
        self.st_rows.append(i)
        self.st_cols.append(j)
        self.st_vals.append(val)

    def st_to_ccs(self):
        """
        将ST格式转换为CCS格式.

        CCS存储:
            ccc[j]: 第j列在icc/acc中的起始索引
            icc[k]: 第k个非零元的行号
            acc[k]: 第k个非零元的值
        """
        nst = len(self.st_vals)
        if nst == 0:
            self.ccc = np.zeros(self.n + 1, dtype=int)
            self.icc = np.array([], dtype=int)
            self.acc = np.array([], dtype=float)
            self.ccs_ready = True
            return

        # 按列排序,再按行排序
        data = list(zip(self.st_cols, self.st_rows, self.st_vals))
        data.sort(key=lambda x: (x[0], x[1]))

        # 统计每列非零元个数并建立ccc
        col_counts = defaultdict(int)
        for col, row, val in data:
            col_counts[col] += 1

        self.ccc = np.zeros(self.n + 1, dtype=int)
        for j in range(1, self.n + 1):
            self.ccc[j] = self.ccc[j - 1] + col_counts.get(j - 1, 0)

        self.icc = np.zeros(nst, dtype=int)
        self.acc = np.zeros(nst, dtype=float)

        # 填充 (允许同一位置的值累加)
        next_pos = self.ccc[:-1].copy()
        for col, row, val in data:
            pos = next_pos[col]
            # 检查是否与上一个位置相同,相同则累加
            if pos > self.ccc[col] and self.icc[pos - 1] == row:
                self.acc[pos - 1] += val
            else:
                self.icc[pos] = row
                self.acc[pos] = val
                next_pos[col] += 1

        # 修正ccc以反映实际存储长度 (处理重复位置压缩后的长度)
        actual_nnz = next_pos[-1] if self.n > 0 else 0
        if actual_nnz < nst:
            self.icc = self.icc[:actual_nnz]
            self.acc = self.acc[:actual_nnz]
            self.ccc[-1] = actual_nnz

        self.ccs_ready = True

    def ccs_mv(self, x):
        """
        CCS格式矩阵-向量乘法: y = A @ x
        """
        if not self.ccs_ready:
            self.st_to_ccs()
        x = np.asarray(x, dtype=float)
        y = np.zeros(self.m, dtype=float)
        for j in range(self.n):
            clo = self.ccc[j]
            chi = self.ccc[j + 1]
            for k in range(clo, chi):
                i = self.icc[k]
                y[i] += self.acc[k] * x[j]
        return y

    def st_mv(self, x):
        """
        ST格式矩阵-向量乘法: y = A @ x
        """
        x = np.asarray(x, dtype=float)
        y = np.zeros(self.m, dtype=float)
        for k in range(len(self.st_vals)):
            i = self.st_rows[k]
            j = self.st_cols[k]
            y[i] += self.st_vals[k] * x[j]
        return y

    def to_dense(self):
        """转为稠密矩阵 (仅用于小规模验证)."""
        A = np.zeros((self.m, self.n), dtype=float)
        for k in range(len(self.st_vals)):
            A[self.st_rows[k], self.st_cols[k]] += self.st_vals[k]
        return A


def generate_room_coupling_graph(n_nodes, connection_prob=0.15, seed=42):
    """
    生成类似USA矩阵的随机稀疏邻接图,模拟房间声学耦合.

    物理意义:
        节点代表空间离散点(或模态),
        边代表声学耦合强度.

    参数:
        n_nodes: 节点数
        connection_prob: 连接概率
        seed: 随机种子

    返回:
        SparseAcousticMatrix 对象
    """
    rng = np.random.default_rng(seed)
    S = SparseAcousticMatrix(n_nodes, n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                S.add_entry(i, j, 1.0)  # 自耦合
            elif rng.random() < connection_prob:
                val = rng.normal(0.0, 0.3)
                S.add_entry(i, j, val)
                S.add_entry(j, i, val)  # 对称耦合
    return S


def acoustic_transfer_matrix_sparse(sensor_positions, source_positions,
                                    k, reflection_coeff=0.8, max_order=2):
    """
    基于镜像源法生成稀疏声学传递矩阵.

    物理模型:
        点源在自由场中的声压:
            p(r) = (j rho0 omega Q / 4pi) * exp(-jkr) / r
        考虑房间反射后,使用镜像源展开:
            p = \sum_{m=0}^{M} R^m * p_m(r_m)
        其中 R 为壁面反射系数, m 为反射阶数.

    参数:
        sensor_positions: (M,3) 传感器坐标数组 [m]
        source_positions: (N,3) 声源坐标数组 [m]
        k: 波数 [rad/m]
        reflection_coeff: 壁面反射系数
        max_order: 最大反射阶数

    返回:
        SparseAcousticMatrix: 复数传递矩阵的模和相位分离存储,
                              此处为简化存储实数幅值矩阵
    """
    # TODO [Hole 2]: 基于镜像源法构建稀疏声学传递矩阵
    # 要求: 计算每个传感器-声源对的欧氏距离r,根据1/r衰减规律计算幅值,
    #      对幅值小于阈值(如0.01)的弱耦合进行截断以保持稀疏性,
    #      使用SparseAcousticMatrix存储结果并返回
    raise NotImplementedError("Hole 2: acoustic_transfer_matrix_sparse 待实现")
