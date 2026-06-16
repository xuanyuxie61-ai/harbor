"""
determinant_qmc.py - 行列式量子蒙特卡洛 (DQMC) 核心采样引擎
=============================================================

科学背景 (Scientific Background):
    行列式量子蒙特卡洛 (Determinant Quantum Monte Carlo, DQMC)
    是求解 Hubbard 模型最严格的数值方法之一 (无近似, 仅有统计误差).

    算法流程:
        1. Trotter 分解: e^{-βH} ≈ [e^{-Δτ H}]^L,  β = L Δτ
        2. HS 变换: 将相互作用分解为辅助场 σ_i(τ) 的二次型
        3. 费米子迹: 对费米子自由度精确积分 → det 表达式
        4. 蒙特卡洛采样: Metropolis-Hastings 更新 σ_i(τ)

    关键量:
        接受率: 由行列式比值决定
            det[M'] / det[M] = det[I + B'_{L} ... B'_0] / det[I + B_{L} ... B_0]
            = det[I + ΔG * (B'_l - B_l)]  (快速更新公式)

融合种子项目:
    - 1235_hrl-team_OptimismPerseveration: 乐观-坚持学习 → 自适应提议分布
    - 798_nested_sequence_display: 嵌套序列 → 嵌套虚时网格

核心公式 (Key Formulas):
    配分函数:
        Z = Σ_{σ} det[M_↑(σ)] det[M_↓(σ)]

    格林函数 (对给定 σ 构型):
        G_{ij}(τ, τ') = -⟨c_i(τ) c†_j(τ')⟩_σ = [M^{-1}]_{(i,τ),(j,τ')}
"""

import numpy as np
from typing import Tuple, Optional, Dict


# ==========================================================================
#  辅助函数: B 矩阵与格林函数 (简化版, 避免循环导入)
# ==========================================================================

def _build_b_matrix_simple(T_mat: np.ndarray, V_diag: np.ndarray,
                           delta_tau: float) -> np.ndarray:
    """简化的 B 矩阵构造 (特征值分解)."""
    Ns = T_mat.shape[0]
    H_eff = T_mat + np.diag(V_diag)
    eigenvalues, eigenvectors = np.linalg.eigh(H_eff)
    exp_vals = np.exp(-delta_tau * eigenvalues)
    exp_vals = np.clip(exp_vals, 1e-300, 1e300)
    return eigenvectors @ np.diag(exp_vals) @ eigenvectors.T


def _compute_green_function(T_mat: np.ndarray,
                            hs_config: np.ndarray,
                            delta_tau: float, U: float
                            ) -> Tuple[np.ndarray, float]:
    """计算等时格林函数 G = [I + B_total]^{-1}."""
    L, Ns = hs_config.shape
    if abs(U) > 1e-15:
        hs_alpha = np.arccosh(np.exp(delta_tau * abs(U) / 2.0))
    else:
        hs_alpha = 0.0

    B_total = np.eye(Ns)
    for l in range(L):
        V_diag = hs_config[l] * hs_alpha * np.sign(U + 1e-15)
        B_l = _build_b_matrix_simple(T_mat, V_diag, delta_tau)
        B_total = B_l @ B_total
        # 定期归一化防止溢出
        norm = np.linalg.norm(B_total, ord='fro')
        if norm > 1e100:
            B_total /= norm

    M = np.eye(Ns) + B_total
    try:
        G = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        G = np.linalg.pinv(M)
    sign_det = float(np.sign(np.linalg.det(M)))
    return G, sign_det


# ==========================================================================
#  DQMC 主循环
# ==========================================================================

class DQMCSimulator:
    """
    行列式量子蒙特卡洛模拟器.

    参数:
        T_mat:          (Ns, Ns) 动能矩阵
        U:              在位库仑排斥
        mu:             化学势
        beta:           逆温度 β = 1/T
        L:              虚时切片数
        n_sweeps:       蒙特卡洛扫掠数
        n_therm:        热化扫掠数
        seed:           随机种子
        hs_field_type:  'discrete' 或 'continuous'
    """

    def __init__(self, T_mat: np.ndarray, U: float, mu: float,
                 beta: float, L: int,
                 n_sweeps: int = 100, n_therm: int = 30,
                 seed: int = 4242,
                 hs_field_type: str = 'discrete'):
        self.T_mat = T_mat
        self.Ns = T_mat.shape[0]
        self.U = U
        self.mu = mu
        self.beta = beta
        self.L = L
        self.delta_tau = beta / L
        self.n_sweeps = n_sweeps
        self.n_therm = n_therm
        self.rng = np.random.default_rng(seed)
        self.hs_field_type = hs_field_type

        # HS 耦合常数
        if abs(U) > 1e-15:
            self.hs_alpha = np.arccosh(np.exp(self.delta_tau * abs(U) / 2.0))
        else:
            self.hs_alpha = 0.0

        # 化学势修正
        self.T_mat_mu = self.T_mat - mu * np.eye(self.Ns)

        # 初始化 HS 场
        self._init_hs_field()

        # 自适应提议分布参数
        self.sigma_prop = 0.5
        self.target_accept = 0.35

    def _init_hs_field(self):
        """初始化 HS 辅助场构型."""
        if self.hs_field_type == 'discrete':
            self.hs_config = self.rng.choice([-1.0, 1.0],
                                              size=(self.L, self.Ns))
        else:
            sigma_max = 4.0
            self.hs_config = self.rng.uniform(-sigma_max, sigma_max,
                                               size=(self.L, self.Ns))

    def _propose_update(self, l: int, i: int) -> Tuple[float, float]:
        """
        提议更新格点 i、虚时片 l 的 HS 场.

        对离散场: σ → -σ (翻转)
        对连续场: σ → σ + δ,  δ ~ N(0, σ_prop²)

        快速更新: 对单格点翻转, det 比值简化为:
            R = 1 + (1 - e^{-2α σ}) [G_{ii} + G_{ii} - 1]

        返回:
            (log_accept_ratio, sigma_new)
        """
        sigma_old = self.hs_config[l, i]

        if self.hs_field_type == 'discrete':
            sigma_new = -sigma_old
        else:
            delta = self.rng.normal(0, self.sigma_prop)
            sigma_new = np.clip(sigma_old + delta, -4.0, 4.0)

        # 计算格林函数
        G, _ = _compute_green_function(self.T_mat_mu, self.hs_config,
                                       self.delta_tau, self.U)

        # 单格点更新的 det 比值
        delta_V = (sigma_new - sigma_old) * self.hs_alpha * np.sign(self.U + 1e-15)
        factor = np.exp(-self.delta_tau * delta_V) - 1.0

        # det 比值 (对单格点更新, 退化为标量)
        # 使用 Sherman-Morrison: det[I + factor * |i⟩⟨i| G] = 1 + factor * G_{ii}
        det_ratio = 1.0 + factor * G[i, i].real

        if abs(det_ratio) < 1e-300:
            log_accept = -100.0
        else:
            # 自旋向上和向下贡献 |det|²
            log_accept = 2.0 * np.log(abs(det_ratio))

        # 连续场的玻色作用量修正
        if self.hs_field_type == 'continuous':
            delta_S_B = 0.5 * (sigma_new**2 - sigma_old**2)
            log_accept -= delta_S_B

        return log_accept, sigma_new

    def sweep(self) -> Dict[str, float]:
        """
        一次蒙特卡洛扫掠: 遍历所有 (l, i) 提议更新.
        """
        n_accept = 0
        n_total = self.L * self.Ns

        for l in range(self.L):
            for i in range(self.Ns):
                log_accept, sigma_new = self._propose_update(l, i)
                if log_accept >= 0.0 or self.rng.uniform() < np.exp(min(log_accept, 0.0)):
                    self.hs_config[l, i] = sigma_new
                    n_accept += 1

        accept_rate = n_accept / max(n_total, 1)

        # 自适应调整提议步长 (融合 1235 的乐观-坚持策略)
        if self.hs_field_type == 'continuous':
            gamma = 0.1 / (1.0 + n_total * 0.001)
            self.sigma_prop += gamma * (self.target_accept - accept_rate)
            self.sigma_prop = max(0.01, min(self.sigma_prop, 5.0))

        # 测量可观测量
        G, sign = _compute_green_function(self.T_mat_mu, self.hs_config,
                                          self.delta_tau, self.U)
        energy = self._measure_energy(G)
        double_occ = self._measure_double_occupancy(G)

        return {
            'accept_rate': accept_rate,
            'sign': sign,
            'energy': energy,
            'double_occupancy': double_occ,
        }

    def _measure_energy(self, G: np.ndarray) -> float:
        """
        测量总能量 E = ⟨H⟩.

        动能:  E_kin = 2 Tr[T G]  (自旋因子 2)
        势能:  E_pot = U Σ_i ⟨n_{i↑}⟩⟨n_{i↓}⟩
               ⟨n_{iσ}⟩ = 1 - G_{ii}
        """
        E_kin = 2.0 * np.trace(self.T_mat @ G).real
        n_sites = np.array([1.0 - G[i, i].real for i in range(self.Ns)])
        E_pot = self.U * np.sum(n_sites ** 2)
        return E_kin + E_pot

    def _measure_double_occupancy(self, G: np.ndarray) -> float:
        """
        测量双占据 D = (1/Ns) Σ_i ⟨n_{i↑} n_{i↓}⟩.

        D 是 Mott 转变的序参量:
            D → 0    (Mott 绝缘体)
            D → 1/4  (金属, 半填充)
        """
        n_sites = np.array([1.0 - G[i, i].real for i in range(self.Ns)])
        return float(np.mean(n_sites ** 2))

    def run(self) -> Dict[str, object]:
        """
        运行完整 DQMC 模拟.

        流程:
            1. 热化阶段: n_therm 次扫掠
            2. 测量阶段: n_sweeps 次扫掠

        返回结果字典.
        """
        # 热化
        for _ in range(self.n_therm):
            self.sweep()

        # 测量
        energy_series = []
        doub_occ_series = []
        sign_series = []
        accept_series = []

        for _ in range(self.n_sweeps):
            meas = self.sweep()
            energy_series.append(meas['energy'])
            doub_occ_series.append(meas['double_occupancy'])
            sign_series.append(meas['sign'])
            accept_series.append(meas['accept_rate'])

        energy_arr = np.array(energy_series)
        doub_occ_arr = np.array(doub_occ_series)
        sign_arr = np.array(sign_series)
        accept_arr = np.array(accept_series)

        n_eff = max(len(energy_arr), 1)
        return {
            'energy_series': energy_arr,
            'double_occ_series': doub_occ_arr,
            'sign_series': sign_arr,
            'accept_series': accept_arr,
            'mean_energy': float(np.mean(energy_arr)),
            'std_energy': float(np.std(energy_arr) / np.sqrt(n_eff)),
            'mean_double_occ': float(np.mean(doub_occ_arr)),
            'std_double_occ': float(np.std(doub_occ_arr) / np.sqrt(n_eff)),
            'mean_sign': float(np.mean(sign_arr)),
            'mean_accept_rate': float(np.mean(accept_arr)),
        }


# ==========================================================================
#  嵌套虚时网格 (融合 798_nested_sequence_display)
# ==========================================================================

class NestedImaginaryTimeGrid:
    """
    嵌套虚时网格: 多尺度虚时离散化.

    融合种子项目 798_nested_sequence_display.
    原项目展示嵌套序列的层次结构 (每层是上层的超集).
    物理映射: 多网格 DQMC 中的虚时轴嵌套细化.

    Level 0:  τ = {0, β}
    Level 1:  τ = {0, β/2, β}
    Level 2:  τ = {0, β/4, β/2, 3β/4, β}
    ...
    Level L:  τ = {0, β/2^L, ..., β}
    """

    def __init__(self, beta: float, max_level: int = 4):
        self.beta = beta
        self.max_level = max_level
        self.grids = {}
        self._build_grids()

    def _build_grids(self):
        """构建各层级的嵌套虚时网格."""
        for level in range(self.max_level + 1):
            n_slices = 2 ** level
            tau = np.linspace(0, self.beta, n_slices + 1)
            self.grids[level] = tau

    def get_grid(self, level: int) -> np.ndarray:
        if level not in self.grids:
            raise ValueError(f"层级 {level} 不存在, max={self.max_level}")
        return self.grids[level]

    def get_common_points(self, level1: int, level2: int) -> np.ndarray:
        """找到两个层级网格的公共点 (嵌套性质保证低层 ⊂ 高层)."""
        g1 = set(np.round(self.grids[level1], 10))
        g2 = set(np.round(self.grids[level2], 10))
        return np.array(sorted(g1 & g2))

    def interpolate_to_fine(self, values_coarse: np.ndarray,
                            level_coarse: int, level_fine: int
                            ) -> np.ndarray:
        """粗网格 → 细网格的线性插值 (多网格初始猜测)."""
        tau_coarse = self.grids[level_coarse]
        tau_fine = self.grids[level_fine]
        return np.interp(tau_fine, tau_coarse, values_coarse)

    def hierarchy_info(self) -> str:
        """返回嵌套层次结构的文本描述."""
        lines = []
        for level in range(self.max_level + 1):
            grid = self.grids[level]
            lines.append(f"Level {level}: {len(grid)} points, "
                         f"dtau = {self.beta / max(len(grid) - 1, 1):.4f}")
        return '\n'.join(lines)
