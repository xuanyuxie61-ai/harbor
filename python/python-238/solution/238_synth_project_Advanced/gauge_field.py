"""
gauge_field.py
==============

SU(3) 规范场 (link 变量) 的数据结构, 提供场运算与并行输运子.

数学定义:
---------
在格点 x 的 μ 方向上的 link 变量为:
    U_μ(x) = exp(i a g A_μ(x + a/2 μ̂))
其中 A_μ 为规范场 (Hermit 无迹矩阵值).

规范变换:
    U_μ(x) → Ω(x) U_μ(x) Ω^†(x + μ̂)
其中 Ω(x) ∈ SU(3).

plaquette (最小规范不变 loop):
    U_μν(x) = U_μ(x) U_ν(x + μ̂) U_μ^†(x + ν̂) U_ν^†(x)

Wilson loop:
    W(R, T) = Tr Π_{l∈∂(R×T)} U_l

Polyakov loop (时间方向的 Wilson line):
    L(x⃗) = Π_{t=0}^{Nt-1} U_4(x⃗, t)
    ⟨L⟩ 为退禁闭序参量:
        ⟨L⟩ = 0   (禁闭相, Z_3 对称)
        ⟨L⟩ ≠ 0   (退禁闭相, Z_3 自发破缺)

本模块融合种子项目:
  - 631_l4lib:   SU(3) 矩阵的 XOR parity 索引
  - 431_filum:   配置文件的顺序命名与读写
  - 545_house:   参考构型 (冷/热启动) 的生成
"""

import numpy as np
from typing import Tuple, Optional
from lattice_geometry import LatticeGeometry
from su3_algebra import (
    project_to_su3, su3_random, color_trace, traceless_anti_hermitian
)


class GaugeField:
    """SU(3) 规范场: 4 个方向 × V 个站点的 link 矩阵集合.

    数据结构:
        U[mu][idx] = 3×3 complex128 矩阵, 属于 SU(3).
        内部以形状 (4, volume, 3, 3) 的连续数组存储.

    约定:
        μ = 0, 1, 2: 空间方向
        μ = 3: 时间方向 (紧致)
    """

    def __init__(self, geometry: LatticeGeometry,
                 initial: str = 'cold',
                 seed: int = 42):
        """
        参数:
            geometry: 格点几何
            initial: 'cold' (单位矩阵) 或 'hot' (随机 SU(3))
            seed: 随机种子 (仅 hot 启动时使用)
        """
        self.geom = geometry
        self.links = np.zeros((4, geometry.volume, 3, 3), dtype=np.complex128)

        if initial == 'cold':
            self._cold_start()
        elif initial == 'hot':
            self._hot_start(seed)
        else:
            raise ValueError(f"未知启动类型: {initial}")

    def _cold_start(self):
        """冷启动: 所有 link 设为单位矩阵.

        对应于零温场 A_μ = 0, 即 β → ∞ 的基态.
        用于验证作用量、力计算等的正确性.
        """
        for mu in range(4):
            for idx in range(self.geom.volume):
                self.links[mu, idx] = np.eye(3, dtype=np.complex128)

    def _hot_start(self, seed: int):
        """热启动: 所有 link 随机采样自 Haar 测度.

        对应于极高温度 (无序) 初始状态, 用于快速越过临界区.
        """
        rng = np.random.default_rng(seed)
        for mu in range(4):
            for idx in range(self.geom.volume):
                self.links[mu, idx] = su3_random(rng)

    # --------------------------------------------------------
    # 基本场运算
    # --------------------------------------------------------

    def get_link(self, mu: int, idx: int) -> np.ndarray:
        """获取 link U_μ(x)."""
        return self.links[mu, idx].copy()

    def get_link_backward(self, mu: int, idx: int) -> np.ndarray:
        """获取反向 link: U_μ^†(x - μ̂)."""
        idx_b = self.geom.neighbor_idx(idx, mu, -1)
        return self.links[mu, idx_b].conj().T

    def staple(self, idx: int, mu: int) -> np.ndarray:
        """计算 link U_μ(x) 的 "staple" 和:

        S_μ(x) = Σ_{ν≠μ} [ U_ν(x+μ̂) U_μ^†(x+ν̂) U_ν^†(x)
                           + U_ν^†(x+μ̂-ν̂) U_μ^†(x-ν̂) U_ν(x-ν̂) ]

        staple 是作用量对 link 的导数的核心:
            S_g = -(β/3) Σ_{x,μ} Re Tr[U_μ(x) S_μ(x)]

        对每个 μ, staple 由 6 个 "上下" 矩形贡献 (ν ≠ μ, 正向 + 反向).
        """
        staple_sum = np.zeros((3, 3), dtype=np.complex128)
        for nu in range(4):
            if nu == mu:
                continue
            # 正向矩形 (upper staple)
            # U_ν(x+μ̂) U_μ^†(x+ν̂) U_ν^†(x)
            idx_mu = self.geom.neighbor_idx(idx, mu, +1)
            idx_nu = self.geom.neighbor_idx(idx, nu, +1)
            u_nu_xmu = self.links[nu, idx_mu]
            u_mu_xnu = self.links[mu, idx_nu]
            u_nu_x = self.links[nu, idx]
            upper = u_nu_xmu @ u_mu_xnu.conj().T @ u_nu_x.conj().T
            staple_sum += upper

            # 反向矩形 (lower staple)
            # U_ν^†(x+μ̂-ν̂) U_μ^†(x-ν̂) U_ν(x-ν̂)
            idx_mu_nu = self.geom.neighbor_idx(idx_mu, nu, -1)
            idx_nu_b = self.geom.neighbor_idx(idx, nu, -1)
            u_nu_dag_xmu_nu = self.links[nu, idx_mu_nu].conj().T
            u_mu_dag_xnu_b = self.links[mu, idx_nu_b].conj().T
            u_nu_xnu_b = self.links[nu, idx_nu_b]
            lower = u_nu_dag_xmu_nu @ u_mu_dag_xnu_b @ u_nu_xnu_b
            staple_sum += lower

        return staple_sum

    def plaquette(self, idx: int, mu: int, nu: int) -> np.ndarray:
        """计算单 plaquette U_μν(x).

        U_μν(x) = U_μ(x) U_ν(x+μ̂) U_μ^†(x+ν̂) U_ν^†(x)
        """
        idx_mu = self.geom.neighbor_idx(idx, mu, +1)
        idx_nu = self.geom.neighbor_idx(idx, nu, +1)
        u_mu_x = self.links[mu, idx]
        u_nu_xmu = self.links[nu, idx_mu]
        u_mu_xnu = self.links[mu, idx_nu]
        u_nu_x = self.links[nu, idx]
        return u_mu_x @ u_nu_xmu @ u_mu_xnu.conj().T @ u_nu_x.conj().T

    def avg_plaquette(self) -> float:
        """计算全格点平均归一化 plaquette:

        P = (1 / (6 V)) Σ_{x,μ<ν} (1/3) Re Tr U_μν(x)

        连续极限: P = 1 - (g^2/6) ⟨F_μν F_μν⟩ + O(a^2)
        微扰展开: P ≈ 1 - c_F g^2 / β + O(g^4)
        """
        p_sum = 0.0
        count = 0
        for idx in range(self.geom.volume):
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    plaq = self.plaquette(idx, mu, nu)
                    p_sum += color_trace(plaq) / 3.0
                    count += 1
        return p_sum / count if count > 0 else 0.0

    # --------------------------------------------------------
    # Wilson line 与 Polyakov loop
    # --------------------------------------------------------

    def polyakov_loop_spatial(self, x_idx: int) -> np.ndarray:
        """计算空间位置 x⃗ 处的 Polyakov loop 矩阵:

        L(x⃗) = Π_{t=0}^{Nt-1} U_4(x⃗, t)

        这是时间方向上的 Wilson line, 在有限温 QCD 中作为
        退禁闭序参量. 其迹的期望值 ⟨(1/3) Tr L⟩ 是
        Z_3 中心对称性的序参量.

        参数:
            x_idx: 空间站点索引 (0, 1, ..., Ns^3 - 1)
        """
        # 将空间索引转为 4D 索引 (t = 0)
        x = x_idx % self.geom.Ns
        y = (x_idx // self.geom.Ns) % self.geom.Ns
        z = x_idx // (self.geom.Ns ** 2)
        idx_4d = self.geom.coord_to_idx(x, y, z, 0)

        L_matrix = np.eye(3, dtype=np.complex128)
        current_idx = idx_4d
        for t in range(self.geom.Nt):
            U_t = self.links[3, current_idx]
            L_matrix = L_matrix @ U_t
            current_idx = self.geom.neighbor_idx(current_idx, 3, +1)
        return L_matrix

    def avg_polyakov_loop(self) -> complex:
        """空间平均 Polyakov loop:

        ⟨L⟩ = (1 / (3 V_s)) Σ_{x⃗} Tr L(x⃗)

        期望值:
            禁闭相: ⟨L⟩ ≈ 0 (Z_3 对称)
            退禁闭相: ⟨L⟩ ≠ 0, 取值在 Z_3 的 3 个中心之一
        """
        L_sum = 0.0 + 0.0j
        for x_idx in range(self.geom.spatial_volume):
            L_matrix = self.polyakov_loop_spatial(x_idx)
            L_sum += np.trace(L_matrix) / 3.0
        return L_sum / self.geom.spatial_volume

    def polyakov_loop_correlator(self, r_sq: int) -> complex:
        """Polyakov loop 关联函数 (静态夸克-反夸克势):

        C(r) = ⟨L(x⃗) L^†(x⃗ + r⃗)⟩

        在大 r 极限下:
            C(r) ∝ exp(-F_qq̄(r) / T)
        其中 F_qq̄ 为静态夸克自由能.

        参数:
            r_sq: 距离平方 (格点单位)
        """
        # 计算所有空间距离为 r 的 Polyakov loop 对
        corr_sum = 0.0 + 0.0j
        count = 0
        for x_idx in range(self.geom.spatial_volume):
            L_x = self.polyakov_loop_spatial(x_idx)
            for y_idx in range(self.geom.spatial_volume):
                if x_idx == y_idx:
                    continue
                # 计算 4D 索引的距离
                x_4d = self.geom.coord_to_idx(
                    x_idx % self.geom.Ns,
                    (x_idx // self.geom.Ns) % self.geom.Ns,
                    x_idx // (self.geom.Ns ** 2),
                    0
                )
                y_4d = self.geom.coord_to_idx(
                    y_idx % self.geom.Ns,
                    (y_idx // self.geom.Ns) % self.geom.Ns,
                    y_idx // (self.geom.Ns ** 2),
                    0
                )
                d_sq = self.geom.distance_sq(x_4d, y_4d)
                if abs(d_sq - r_sq) < 1e-10:
                    L_y = self.polyakov_loop_spatial(y_idx)
                    corr_sum += np.trace(L_x @ L_y.conj().T) / 3.0
                    count += 1
        if count == 0:
            return 0.0 + 0.0j
        return corr_sum / count

    # --------------------------------------------------------
    # 规范变换
    # --------------------------------------------------------

    def gauge_transform(self, omega: np.ndarray):
        """应用局域规范变换:

        U_μ(x) → Ω(x) U_μ(x) Ω^†(x + μ̂)

        参数:
            omega: shape = (volume, 3, 3), 每个站点的 SU(3) 矩阵
        """
        new_links = np.zeros_like(self.links)
        for mu in range(4):
            for idx in range(self.geom.volume):
                idx_mu = self.geom.neighbor_idx(idx, mu, +1)
                new_links[mu, idx] = (
                    omega[idx]
                    @ self.links[mu, idx]
                    @ omega[idx_mu].conj().T
                )
        self.links = new_links

    def random_gauge_transform(self, seed: Optional[int] = None):
        """随机规范变换, 用于检测物理量是否规范不变."""
        rng = np.random.default_rng(seed)
        omega = np.zeros((self.geom.volume, 3, 3), dtype=np.complex128)
        for idx in range(self.geom.volume):
            omega[idx] = su3_random(rng)
        self.gauge_transform(omega)

    # --------------------------------------------------------
    # 场更新 (用于 HMC / Heatbath)
    # --------------------------------------------------------

    def update_link(self, mu: int, idx: int, delta: np.ndarray):
        """将 link 乘以代数元素: U → exp(δ) U.

        参数:
            delta: su(3) 代数元素 (反 Hermit 无迹矩阵)
        """
        from su3_algebra import su3_exp_safe
        exp_delta = su3_exp_safe(delta)
        self.links[mu, idx] = project_to_su3(exp_delta @ self.links[mu, idx])

    def momenta(self) -> np.ndarray:
        """分配共轭动量 P_μ(x) ∈ su(3).

        在 HMC 中, 动量为 8 个实自由度 (对应 SU(3) 的 8 个生成元),
        以反 Hermit 无迹矩阵存储.
        """
        return np.zeros((4, self.geom.volume, 3, 3), dtype=np.complex128)

    def sample_momenta(self, seed: Optional[int] = None) -> np.ndarray:
        """从热分布采样动量:

        H_kinetic = (1/2) Σ_{x,μ} Tr(P_μ^2)
        各分量独立高斯分布: P_μ(x) = i Σ_a p_a T_a, p_a ~ N(0, 1)
        """
        from su3_algebra import GENERATORS
        rng = np.random.default_rng(seed)
        P = np.zeros((4, self.geom.volume, 3, 3), dtype=np.complex128)
        for mu in range(4):
            for idx in range(self.geom.volume):
                coeffs = rng.standard_normal(8)
                P_mu_idx = np.zeros((3, 3), dtype=np.complex128)
                for a in range(8):
                    P_mu_idx += coeffs[a] * GENERATORS[a]
                P[mu, idx] = 1j * P_mu_idx
        return P

    def kinetic_energy(self, P: np.ndarray) -> float:
        """动量的动能:

        H_kin = (1/2) Σ_{x,μ} Tr(P_μ(x) P_μ^†(x))
              = (1/2) Σ_{x,μ,a} (p_a^{μ,x})^2
        """
        kinetic = 0.0
        for mu in range(4):
            for idx in range(self.geom.volume):
                kinetic += np.trace(P[mu, idx] @ P[mu, idx].conj().T).real
        return 0.5 * kinetic

    # --------------------------------------------------------
    # 数值稳定性: unitarity 偏差
    # --------------------------------------------------------

    def unitarity_deviation(self) -> float:
        """监测 link 矩阵偏离 SU(3) 的程度:

        δ = max_{x,μ} || U_μ^†(x) U_μ(x) - I ||_F

        在 HMC 演化中, 数值误差会使 unitarity 逐渐丢失,
        需要定期重投影或限制步长.
        """
        max_dev = 0.0
        for mu in range(4):
            for idx in range(self.geom.volume):
                uhu = self.links[mu, idx].conj().T @ self.links[mu, idx]
                dev = np.linalg.norm(uhu - np.eye(3), 'fro')
                max_dev = max(max_dev, dev)
        return max_dev
