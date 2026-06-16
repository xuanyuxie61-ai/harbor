"""
sparse_decay_io.py
------------------
稀疏衰变链转移矩阵的 CRS 存储与输入-输出型分支比网络分析。
映射自种子项目 228_crs_io (CRS 稀疏矩阵读写) 和
1155_anhkiet...EEIO-Analysis (投入产出分析)。

物理动机:
  在复杂衰变链 (如 b → c → s 级联) 或 B 介子强子化后多粒子末态中,
  每个中间态到各个末态的分支比构成一个转移矩阵 B_{ij}:
      n_j^{final} = Σ_i B_{ij} n_i^{initial}
  对多代级联:
      n^{(k)} = B^k n^{(0)}
  当中间态数目很大 (数十到数百) 时, 矩阵 B 高度稀疏 (每行非零元
  仅为该粒子实际开放的衰变道数, 通常 2~8 个)。使用 CRS 格式存储。

  借鉴投入产出分析 (Leontief IO) 框架:
      x = A x + y   →   x = (I - A)^{-1} y
  其中 A_{ij} 为部门 j 对部门 i 的输入系数, x 为总产出, y 为最终需求。
  在 B 衰变中, 类似地:
      n = B n + f   →   n = (I - B)^{-1} f
  其中 n_i 为粒子 i 的稳态数目, B_{ij} 为 j → i + ... 的部分分支比,
  f_i 为外部源项 (例如 initial B production)。

本模块实现:
  1) CRS 稀疏矩阵的构造、读写、矩阵-向量乘法 (仿 228);
  2) 基于 Leontief 逆 (I - B)^{-1} 的衰变链稳态分析 (仿 1155);
  3) 碳税场景类比: 不同 "价格" (能量阈值) 下分支比的灵敏度分析。
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

import numpy as np


# ========================================================================== #
#                    CRS 稀疏矩阵类 (仿 228_crs_io)                        #
# ========================================================================== #
class SparseCRS:
    """
    Compressed Row Storage 稀疏矩阵:
      - row_ptr: length N+1, 第 i 行的非零元从 row_ptr[i] 到 row_ptr[i+1]-1
      - col_idx: length NNZ, 非零元的列索引
      - values:  length NNZ, 非零元的值
    构造方法: 从 dict-of-dict {i: {j: val}} 转换。
    """
    def __init__(
            self,
            row_ptr: np.ndarray,
            col_idx: np.ndarray,
            values: np.ndarray,
            n: int,
    ):
        self.row_ptr = np.asarray(row_ptr, dtype=int)
        self.col_idx = np.asarray(col_idx, dtype=int)
        self.values  = np.asarray(values, dtype=float)
        self.n = n
        self.nnz = len(self.values)
        # 一致性检查
        if len(self.row_ptr) != n + 1:
            raise ValueError(
                f"row_ptr 长度应为 n+1={n+1}, 实际 {len(self.row_ptr)}"
            )
        if self.row_ptr[0] != 0:
            raise ValueError("row_ptr[0] 必须为 0")
        if self.row_ptr[-1] != self.nnz:
            raise ValueError(
                f"row_ptr[-1]={self.row_ptr[-1]} 应等于 nnz={self.nnz}"
            )

    @classmethod
    def from_dict(cls, d: Dict[int, Dict[int, float]], n: int) -> "SparseCRS":
        """从 dict-of-dict 构造 CRS 矩阵。"""
        row_ptr = [0]
        col_idx = []
        values = []
        for i in range(n):
            row_data = d.get(i, {})
            for j in sorted(row_data.keys()):
                col_idx.append(j)
                values.append(row_data[j])
            row_ptr.append(len(col_idx))
        return cls(np.array(row_ptr), np.array(col_idx),
                   np.array(values, dtype=float), n)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """矩阵-向量乘法 y = A x (CRS 格式)。"""
        if len(x) != self.n:
            raise ValueError(f"向量长度 {len(x)} 不等于矩阵维度 {self.n}")
        y = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            s = 0.0
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                s += self.values[k] * x[self.col_idx[k]]
            y[i] = s
        return y

    def density(self) -> float:
        """稀疏度: 非零元比例。"""
        return self.nnz / (self.n * self.n) if self.n > 0 else 0.0

    def to_dense(self) -> np.ndarray:
        """转换为稠密 numpy 数组 (用于小矩阵)。"""
        A = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                A[i, self.col_idx[k]] = self.values[k]
        return A


def crs_write(prefix: str, mat: SparseCRS) -> None:
    """
    将 CRS 矩阵写入 3 个文件 (仿 228_crs_io):
      - {prefix}_row.txt: row_ptr
      - {prefix}_col.txt: col_idx
      - {prefix}_val.txt: values
    """
    np.savetxt(f"{prefix}_row.txt", mat.row_ptr, fmt="%d")
    np.savetxt(f"{prefix}_col.txt", mat.col_idx, fmt="%d")
    np.savetxt(f"{prefix}_val.txt", mat.values, fmt="%.15e")


def crs_read(prefix: str) -> SparseCRS:
    """从文件读回 CRS 矩阵 (仿 228_crs_io)。"""
    row_ptr = np.loadtxt(f"{prefix}_row.txt", dtype=int)
    col_idx = np.loadtxt(f"{prefix}_col.txt", dtype=int)
    values  = np.loadtxt(f"{prefix}_val.txt", dtype=float)
    n = len(row_ptr) - 1
    return SparseCRS(row_ptr, col_idx, values, n)


# ========================================================================== #
#         B 衰变链转移矩阵构造                                              #
# ========================================================================== #
# 简化的 B 衰变道列表: (母粒子, 子粒子列表, 分支比)
# 数据参考 PDG 2024 (仅为教学示例, 非完整)
B_DECAY_CHANNELS = [
    # B0 衰变
    ("B0", ["Dp", "pim"],       0.00254),   # B0 → D+ pi-
    ("B0", ["Dm", "pip"],       0.00254),   # B0 → D- pi+
    ("B0", ["Dstar_m", "pip"],  0.00489),
    ("B0", ["Jpsi", "KS"],      0.00087),   # 黄金道
    ("B0", ["pip", "pim"],      0.0000005), # 罕见
    # B+ 衰变
    ("Bp", ["D0bar", "pip"],    0.0049),
    ("Bp", ["D0", "pip"],       0.00005),
    ("Bp", ["Jpsi", "Kp"],      0.00103),
    # D 介子衰变
    ("Dp",  ["KS", "pip"],      0.0140),
    ("Dp",  ["pi0", "pip"],     0.0017),
    ("Dm",  ["K0bar", "pim"],   0.0140),
    ("D0",  ["Km", "pip"],      0.0389),
    ("D0bar", ["Kp", "pim"],    0.0389),
    # K 介子衰变 (长寿命)
    ("KS",  ["pip", "pim"],     0.6920),
    ("KS",  ["pi0", "pi0"],     0.3069),
    ("Kp",  ["pip", "pi0"],     0.2066),
    ("Km",  ["pim", "pi0"],     0.2066),
]

PARTICLE_INDEX = {
    "B0": 0, "Bp": 1, "Bm": 2, "Bs": 3,
    "Dp": 4, "Dm": 5, "D0": 6, "D0bar": 7, "Dstar_m": 8,
    "pip": 9, "pim": 10, "pi0": 11,
    "Kp": 12, "Km": 13, "KS": 14, "K0bar": 15,
    "Jpsi": 16,
}
N_PARTICLES = len(PARTICLE_INDEX)


def build_decay_transition_matrix(
        channels: List[Tuple[str, List[str], float]] = None,
        include_stable_leptons: bool = False,
) -> SparseCRS:
    """
    构造衰变链转移矩阵 B, 其中 B_{ij} 为粒子 j → 粒子 i + ... 的部分分支比。
    矩阵按列归一: Σ_i B_{ij} <= 1 (每个母粒子的总分支比 ≤ 1)。
    """
    if channels is None:
        channels = B_DECAY_CHANNELS
    d: Dict[int, Dict[int, float]] = {i: {} for i in range(N_PARTICLES)}
    for (parent, daughters, br) in channels:
        if parent not in PARTICLE_INDEX:
            continue
        j = PARTICLE_INDEX[parent]
        for daug in daughters:
            if daug not in PARTICLE_INDEX:
                continue
            i = PARTICLE_INDEX[daug]
            # 部分分支比: 每个 daughter 平分母粒子的 BR
            partial = br / max(len(daughters), 1)
            d[i][j] = d[i].get(j, 0.0) + partial
    return SparseCRS.from_dict(d, N_PARTICLES)


# ========================================================================== #
#          投入-产出型衰变链稳态分析 (仿 1155_EEIO)                        #
# ========================================================================== #
def leontief_inverse(
        B: SparseCRS,
        regularization: float = 1.0e-10,
) -> np.ndarray:
    """
    计算 Leontief 逆矩阵 L = (I - B)^{-1}。
    物理意义: L_{ij} 表示注入 1 个 j 粒子最终产生的 i 粒子总数
    (含级联衰变贡献)。

    使用小正则化 epsilon 避免奇异性:
        L = (I - B + epsilon I)^{-1}
    对物理解, epsilon 应远小于 1 - rho(B), 其中 rho(B) 为谱半径。
    """
    I = np.eye(B.n, dtype=float)
    A = I - B.to_dense() + regularization * I
    return np.linalg.inv(A)


def decay_chain_cascade(
        initial_population: np.ndarray,
        B: SparseCRS,
        n_generation: int = 5,
) -> List[np.ndarray]:
    """
    计算级联衰变 n 代后的粒子数目分布:
        n^{(k+1)} = B n^{(k)}
    返回 [n^{(0)}, n^{(1)}, ..., n^{(n_generation)}]
    """
    history = [initial_population.copy()]
    n_k = initial_population.copy().astype(float)
    for _ in range(n_generation):
        n_k = B.matvec(n_k)
        history.append(n_k.copy())
    return history


def cascade_steady_state(
        source: np.ndarray,
        B: SparseCRS,
) -> np.ndarray:
    """
    计算稳态分布:
        n* = (I - B)^{-1} source
    物理含义: 在恒定源注入下, 系统达到的平衡分布。
    """
    L = leontief_inverse(B)
    return L @ source


def sensitivity_to_branch_ratio(
        B: SparseCRS,
        source: np.ndarray,
        channel_idx: int,
        delta_br: float,
) -> Tuple[float, float]:
    """
    分析分支比扰动对稳态分布的灵敏度 (仿 1155 的碳税场景):
      1) 计算基线稳态 n0 = (I - B)^{-1} source;
      2) 对第 channel_idx 个衰变道的分支比扰动 delta_br;
      3) 重新计算稳态 n1;
      4) 返回 (基线总产额, 相对变化率)。
    灵敏度 = (Σ_i |n1_i - n0_i|) / (Σ_i n0_i) / |delta_br|
    """
    n0 = cascade_steady_state(source, B)
    total0 = n0.sum()
    # 扰动: 复制 B 并修改一个非零元
    values_new = B.values.copy()
    if channel_idx < 0 or channel_idx >= len(values_new):
        raise ValueError(f"channel_idx {channel_idx} 超出范围")
    values_new[channel_idx] += delta_br
    B_new = SparseCRS(B.row_ptr.copy(), B.col_idx.copy(), values_new, B.n)
    n1 = cascade_steady_state(source, B_new)
    total1 = n1.sum()
    rel_change = (total1 - total0) / max(total0, 1.0e-30)
    if abs(delta_br) < 1.0e-30:
        return float(total0), 0.0
    sensitivity = rel_change / abs(delta_br)
    return float(total0), float(sensitivity)


def total_particle_yield(
        source: np.ndarray,
        B: SparseCRS,
        target_idx: int,
) -> float:
    """
    计算给定源分布下, 目标粒子的总产额 (含级联贡献):
        yield = ((I - B)^{-1} source)_{target_idx}
    """
    n_star = cascade_steady_state(source, B)
    return float(n_star[target_idx])
