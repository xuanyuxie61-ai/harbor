# -*- coding: utf-8 -*-
"""
combinatorial_modes.py
======================

组合波模式分析模块.

本模块处理波-粒子相互作用中的模式识别和耦合网络:
  - 累积优势经纪人模型 (mannbach 项目 -> L 壳层耦合网络)
  - 波模式组合枚举 (candy_count 项目 -> 共振模式组合)

物理背景:

1. 波-粒子共振条件:
   对于磁力镜像捕获的粒子, 共振条件为:
     omega - k_parallel * v_parallel = n * Omega_ce / gamma

   其中 n 为谐波数 (n = 0: 朗道共振, n = ±1: 回旋共振, ...).

2. L 壳层耦合网络:
   不同 L 壳层的粒子通过波-粒子相互作用耦合:
     df_L/dt = sum_{L'} W_{LL'} * f_{L'} - Gamma_L * f_L

   其中 W_{LL'} 为耦合权重矩阵, Gamma_L 为损失率.

3. 累积优势 (Cumulative Advantage):
   在网络动力学中, 高连通度的节点更容易获得新的连接:
     P(k_new -> i) ~ k_i^alpha

   在磁层中, 对应于强扩散区域的粒子更容易被加速.

参考文献:
  [1] Mannbach, T. et al., "Cumulative advantage in networks" (2023)
  [2] Horne, R.B. & Thorne, R.M., "Potential waves for relativistic
      electron scattering", GRL (2003)
"""

import numpy as np
import physical_constants as pc


# =============================================================================
#  共振条件计算
# =============================================================================

def cyclotron_resonance_condition(n_harmonic, B, E_MeV, alpha_eq=0.0):
    """
    计算回旋共振条件.

    物理公式:
      omega - k_parallel * v_parallel = n * Omega_ce / gamma

    对于场向传播 (k_perp = 0):
      k_parallel = (n*Omega_ce/gamma - omega) / v_parallel

    参数
    ----
    n_harmonic : int
        谐波数 (n = 0: 朗道, n = ±1: 基频, ...)
    B : float
        磁场强度 [T]
    E_MeV : float
        电子动能 [MeV]
    alpha_eq : float
        赤道投掷角 [rad]

    返回
    -------
    k_parallel : float
        平行波数 [1/m]
    omega_res : float
        共振频率 [rad/s]
    """
    gamma = pc.kinetic_to_lorentz(E_MeV)
    v = pc.C_LIGHT * np.sqrt(1.0 - 1.0/gamma**2)
    v_parallel = v * np.cos(alpha_eq)

    Omega_ce = pc.gyrofrequency(B)

    # 假设 omega << n*Omega_ce/gamma (低频近似)
    omega_res = n_harmonic * Omega_ce / gamma
    if np.abs(v_parallel) > pc.EPSILON_NUM:
        k_parallel = (n_harmonic * Omega_ce / gamma) / v_parallel
    else:
        k_parallel = 0.0

    return k_parallel, omega_res


def enumerate_resonance_modes(n_max=3):
    """
    枚举可能的共振模式组合.

    对于 chorus 波, 可能的共振包括:
      - 朗道共振 (n=0)
      - 反常回旋共振 (n=-1)
      - 正常回旋共振 (n=+1)
      - 高次谐波 (n=±2, ±3, ...)

    参数
    ----
    n_max : int
        最大谐波数

    返回
    -------
    modes : list of dict
        共振模式列表
    """
    modes = []

    # 波模式
    wave_types = ['chorus', 'hiss', 'emice', 'z_mode']

    for wave in wave_types:
        for n in range(-n_max, n_max + 1):
            mode = {
                'wave': wave,
                'harmonic': n,
                'name': f"{wave}_n{n}",
                'type': 'landau' if n == 0 else ('anomalous' if n < 0 else 'cyclotron'),
            }
            modes.append(mode)

    return modes


# =============================================================================
#  L 壳层耦合网络 (来自 cumulative_advantage_brokerage 项目)
# =============================================================================

class LShellCouplingNetwork:
    """
    L 壳层耦合网络.

    节点: L 壳层 (离散化)
    边: 波-粒子相互作用耦合
    权重: 耦合强度 W_{LL'}

    累积优势动力学:
      dW_{ij}/dt = eta * (k_i * k_j)^alpha / sum_{kl} (k_k * k_l)^alpha

    其中 k_i 为节点 i 的度.

    参数
    ----
    L_values : ndarray
        L 壳层值
    alpha : float
        累积优势指数 (0 = 随机, 1 = 线性优先)
    """

    def __init__(self, L_values, alpha=1.0):
        self.L_values = np.asarray(L_values)
        self.n_L = len(L_values)
        self.alpha = alpha

        # 初始化邻接矩阵 (基于距离的初始耦合)
        self.adjacency = self._initialize_adjacency()
        self.degree = np.sum(self.adjacency, axis=1)

    def _initialize_adjacency(self):
        """
        初始化邻接矩阵.

        基于 L 壳层距离的初始耦合:
          W_{ij} ~ exp(-|L_i - L_j| / L_0)
        """
        L0 = 1.0  # 耦合衰减尺度
        L_i, L_j = np.meshgrid(self.L_values, self.L_values, indexing='ij')
        W = np.exp(-np.abs(L_i - L_j) / L0)
        np.fill_diagonal(W, 0.0)
        return W

    def evolve_cumulative_advantage(self, n_steps=100, eta=0.01):
        """
        演化累积优势动力学.

        参数
        ----
        n_steps : int
            演化步数
        eta : float
            学习率

        返回
        -------
        adjacency_history : list of ndarray
            邻接矩阵历史
        """
        W = self.adjacency.copy()
        history = [W.copy()]

        for step in range(n_steps):
            # 计算度
            k = np.sum(W, axis=1)
            k = np.maximum(k, pc.EPSILON_NUM)

            # 累积优势: 优先连接高度节点
            k_i, k_j = np.meshgrid(k, k, indexing='ij')
            preference = (k_i * k_j)**self.alpha
            preference_sum = np.sum(preference)

            if preference_sum > pc.EPSILON_NUM:
                preference /= preference_sum

            # 更新权重
            dW = eta * (preference - W / max(np.sum(W), pc.EPSILON_NUM))
            W += dW
            W = np.maximum(W, 0.0)
            np.fill_diagonal(W, 0.0)

            if step % 20 == 0:
                history.append(W.copy())

        self.adjacency = W
        self.degree = np.sum(W, axis=1)
        return history

    def compute_diffusion_coupling(self):
        """
        计算径向扩散耦合矩阵.

        物理公式:
          Gamma_{ij} = W_{ij} * sqrt(D_{LL,i} * D_{LL,j}) / |L_i - L_j|

        返回
        -------
        Gamma : ndarray
            耦合矩阵
        """
        W = self.adjacency
        L_i, L_j = np.meshgrid(self.L_values, self.L_values, indexing='ij')
        dL = np.abs(L_i - L_j)
        dL = np.maximum(dL, pc.EPSILON_NUM)
        np.fill_diagonal(dL, 1.0)  # 防止除零

        # 简化: 使用单位扩散系数
        D_i = np.ones(self.n_L)
        D_j = D_i[:, np.newaxis]
        D_i = D_i[:, np.newaxis]

        Gamma = W * np.sqrt(D_i * D_j) / dL
        np.fill_diagonal(Gamma, 0.0)

        return Gamma

    def network_diagnostics(self):
        """计算网络诊断信息."""
        n_edges = np.sum(self.adjacency > 0.01)
        max_degree = np.max(self.degree)
        mean_degree = np.mean(self.degree)
        clustering = self._compute_clustering()

        return {
            'n_nodes': self.n_L,
            'n_edges': int(n_edges),
            'max_degree': float(max_degree),
            'mean_degree': float(mean_degree),
            'clustering_coefficient': float(clustering),
            'density': float(n_edges / max(self.n_L * (self.n_L - 1), 1)),
        }

    def _compute_clustering(self):
        """计算平均聚类系数."""
        clustering = 0.0
        for i in range(self.n_L):
            neighbors = np.where(self.adjacency[i, :] > 0.01)[0]
            k_i = len(neighbors)
            if k_i < 2:
                continue
            # 邻居间的连接数
            n_links = 0
            for j_idx in range(len(neighbors)):
                for k_idx in range(j_idx + 1, len(neighbors)):
                    j, k = neighbors[j_idx], neighbors[k_idx]
                    if self.adjacency[j, k] > 0.01:
                        n_links += 1
            clustering += 2.0 * n_links / (k_i * (k_i - 1))

        return clustering / max(self.n_L, 1)


# =============================================================================
#  自检验证
# =============================================================================

def self_test():
    """自检验证."""
    print("=" * 60)
    print("组合波模式分析模块自检验证")
    print("=" * 60)

    # 共振条件
    print("\n--- 共振条件 ---")
    B = pc.dipole_field_magnitude(4.0)
    k_par, omega_res = cyclotron_resonance_condition(1, B, 1.0)
    print(f"  L=4, E=1MeV, n=1: k_par = {k_par:.4e} /m, omega = {omega_res:.4e} rad/s")

    # 共振模式
    modes = enumerate_resonance_modes(n_max=2)
    print(f"  共振模式数: {len(modes)}")

    # L 壳层耦合网络
    print("\n--- L 壳层耦合网络 ---")
    L = np.linspace(2, 7, 11)
    net = LShellCouplingNetwork(L, alpha=1.0)
    history = net.evolve_cumulative_advantage(n_steps=50)
    diag = net.network_diagnostics()
    print(f"  节点数: {diag['n_nodes']}, 边数: {diag['n_edges']}")
    print(f"  聚类系数: {diag['clustering_coefficient']:.4f}")

    return True


if __name__ == "__main__":
    self_test()
