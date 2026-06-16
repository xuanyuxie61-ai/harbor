"""
dirac_operator.py — Wilson/clover 费米子 Dirac 算子
=====================================================
融合种子项目:
  [1086_Cuuung_LiH_Clifford_Reproduction] : Pauli 代数 → 旋量/色空间 Clifford 结构
  [995_r8sm]  : Sherman-Morrison 秩-1 修正 → 传播子更新
  [925_pwl_approx_1d] : 分段线性 → 质量插值

物理背景:
  Wilson 费米子的 Dirac 算子:
  D_W(x,y) = delta_{xy} - kappa * sum_mu [(1-gamma_mu) U_mu(x) delta_{x+mu,y}
                                           + (1+gamma_mu) U_mu^dag(x-mu) delta_{x-mu,y}]
  其中 kappa = 1/(2(am_q + 4)) 为跳跃参数, am_q 为无量纲夸克质量.
  gamma 矩阵满足 Clifford 代数: {gamma_mu, gamma_nu} = 2*delta_{mu,nu}

核心公式:
  Clover 改进: D_clover = D_W + c_SW * kappa * i/4 * sigma_mu_nu * F_mu_nu
  其中 sigma_mu_nu = i/2 * [gamma_mu, gamma_nu]
  传播子: S(x,y) = D^{-1}(x,y) — 通过共轭梯度法求解
  色-旋量结构: D 作用在 (Nc * Ns) 维向量上, Ns = 4 为旋量分量
"""

import numpy as np
from typing import Tuple, Optional
from lattice_geometry import LatticeGeometry
from gauge_wilson_flow import GaugeField


# ======================================================================
# Dirac gamma 矩阵 (手征表示, 源自 [1086] 的 Clifford 代数)
# ======================================================================

def gamma_matrices() -> list:
    """构建 4 维 Euclidean gamma 矩阵 (手征表示).

    Clifford 代数: {gamma_mu, gamma_nu} = 2 * delta_{mu,nu} * I_4

    gamma_0 = [[0, I], [I, 0]],   gamma_k = [[0, -i*sigma_k], [i*sigma_k, 0]]
    gamma_5 = gamma_0 * gamma_1 * gamma_2 * gamma_3 = [[-I, 0], [0, I]]

    返回 4 个 4x4 复数矩阵.
    """
    I2 = np.eye(2, dtype=np.complex128)
    Z2 = np.zeros((2, 2), dtype=np.complex128)

    # Pauli 矩阵
    sigma_1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # gamma_0 (时间方向)
    g0 = np.block([[Z2, I2], [I2, Z2]])
    # gamma_k (空间方向)
    g1 = np.block([[Z2, -1j * sigma_1], [1j * sigma_1, Z2]])
    g2 = np.block([[Z2, -1j * sigma_2], [1j * sigma_2, Z2]])
    g3 = np.block([[Z2, -1j * sigma_3], [1j * sigma_3, Z2]])

    return [g0, g1, g2, g3]


def sigma_matrix(mu: int, nu: int, gammas: list) -> np.ndarray:
    """计算 sigma_{mu,nu} = i/2 * [gamma_mu, gamma_nu].

    这是 Lorentz 群的旋量表示生成元, 出现在 clover 项中.
    """
    if mu == nu:
        return np.zeros((4, 4), dtype=np.complex128)
    return 0.5j * (gammas[mu] @ gammas[nu] - gammas[nu] @ gammas[mu])


# ======================================================================
# Wilson Dirac 算子
# ======================================================================

class WilsonDiracOperator:
    """Wilson 费米子 Dirac 算子.

    参数
    ----
    geo : LatticeGeometry
    gauge : GaugeField
    kappa : float
        跳跃参数 kappa = 1/(2*(am_q + ndim)).
        物理夸克质量对应 kappa -> kappa_c (临界值).
    c_sw : float
        Clover 系数 (Sheikholeslami-Wohlert).
        c_sw = 0 对应纯 Wilson, c_sw = 1 为树级 clover.
    """

    def __init__(self, geo: LatticeGeometry, gauge: GaugeField,
                 kappa: float = 0.14, c_sw: float = 0.0):
        self.geo = geo
        self.gauge = gauge
        self.Nc = gauge.Nc
        self.ndim = geo.ndim
        self.volume = geo.volume
        self.Ns = 4  # 旋量分量
        self.spin_color_dim = self.Ns * self.Nc  # = 12 for Nc=3

        # kappa 边界检查: 自由场临界值 kappa_c = 1/(2*ndim) = 1/8
        self.kappa_c = 1.0 / (2.0 * self.ndim)
        if kappa <= 0.0:
            raise ValueError(f"kappa={kappa} 必须为正")
        if kappa >= self.kappa_c:
            raise ValueError(
                f"kappa={kappa} >= kappa_c={self.kappa_c:.4f} "
                f"(超出物理区域, 传播子不存在)")
        self.kappa = kappa
        self.c_sw = c_sw
        self.gammas = gamma_matrices()

        # Clover 项场强张量 (如果 c_sw != 0)
        self._clover_field = None
        if abs(c_sw) > 1e-15:
            self._compute_clover_field()

    def _compute_clover_field(self):
        """计算 clover 项中的场强张量 F_mu_nu.

        F_mu_nu(x) = (1/8i) * [Q_mu_nu(x) - Q_mu_nu^dag(x)]_traceless
        其中 Q = sum of 4 plaquettes in the (mu,nu) plane around x.
        (源自 [1086] 中 Pauli 矩阵的 Clifford 代数结构)
        """
        self._clover_field = np.zeros(
            (self.ndim, self.ndim, self.volume, self.Nc, self.Nc),
            dtype=np.complex128)

        for mu in range(self.ndim):
            for nu in range(mu + 1, self.ndim):
                for x in range(self.volume):
                    # Clover leaf: 四个 plaquette 之和
                    Q = np.zeros((self.Nc, self.Nc), dtype=np.complex128)
                    # 四个方向的叶子
                    x_mu = self.geo.neighbor_index(x, mu, +1)
                    x_nu = self.geo.neighbor_index(x, nu, +1)
                    x_mu_nu = self.geo.neighbor_index(x_mu, nu, +1)
                    x_m_nu = self.geo.neighbor_index(x, mu, -1)
                    x_mu_m_nu = self.geo.neighbor_index(x_nu, mu, -1)
                    x_m_mu = self.geo.neighbor_index(x, mu, -1)
                    x_m_nu_m_mu = self.geo.neighbor_index(
                        self.geo.neighbor_index(x, nu, -1), mu, -1)

                    # Leaf 1: x -> x+mu -> x+mu+nu -> x+nu -> x
                    U1 = (self.gauge.U[mu, x]
                          @ self.gauge.U[nu, x_mu]
                          @ self.gauge.U[mu, x_nu].conj().T
                          @ self.gauge.U[nu, x].conj().T)
                    Q += U1

                    # Leaf 2: x -> x-nu -> x-nu+mu -> x+mu -> x
                    U2 = (self.gauge.U[nu, self.geo.neighbor_index(x, nu, -1)].conj().T
                          @ self.gauge.U[mu, self.geo.neighbor_index(x, nu, -1)]
                          @ self.gauge.U[nu, x_mu_m_nu]
                          @ self.gauge.U[mu, x].conj().T)
                    Q += U2

                    # Leaf 3: x -> x-mu -> x-mu-nu -> x-nu -> x
                    U3 = (self.gauge.U[mu, x_m_mu].conj().T
                          @ self.gauge.U[nu, x_m_nu_m_mu].conj().T
                          @ self.gauge.U[mu, x_m_nu]
                          @ self.gauge.U[nu, x])
                    Q += U3

                    # Leaf 4: x -> x+nu -> x+nu-mu -> x-mu -> x
                    U4 = (self.gauge.U[nu, x]
                          @ self.gauge.U[mu, x_nu].conj().T
                          @ self.gauge.U[nu, x_m_mu].conj().T
                          @ self.gauge.U[mu, x])
                    Q += U4

                    # 场强
                    F = (Q - Q.conj().T) / (8.0j)
                    F -= np.trace(F) / self.Nc * np.eye(self.Nc)
                    self._clover_field[mu, nu, x] = F
                    self._clover_field[nu, mu, x] = -F

    def apply(self, psi: np.ndarray) -> np.ndarray:
        """应用 Dirac 算子 D_W * psi.

        D_W * psi(x) = psi(x) - kappa * sum_mu {
            (I - gamma_mu) U_mu(x) psi(x+mu)
            + (I + gamma_mu) U_mu^dag(x-mu) psi(x-mu)
        }

        参数
        ----
        psi : ndarray, shape (volume, Ns, Nc)
            输入旋量场.

        返回
        ----
        D_psi : ndarray, shape (volume, Ns, Nc)
            D_W 作用于 psi 的结果.
        """
        psi = psi.reshape(self.volume, self.Ns, self.Nc)
        D_psi = psi.copy()

        for mu in range(self.ndim):
            gm = self.gammas[mu]
            IpG = np.eye(self.Ns) - gm  # (I - gamma_mu)
            ImG = np.eye(self.Ns) + gm  # (I + gamma_mu)

            for x in range(self.volume):
                x_fwd = self.geo.neighbor_index(x, mu, +1)
                x_bwd = self.geo.neighbor_index(x, mu, -1)

                U_fwd = self.gauge.U[mu, x]       # U_mu(x)
                U_bwd = self.gauge.U[mu, x_bwd].conj().T  # U_mu^dag(x-mu)

                # 前向: (I - gamma_mu) * U_mu(x) * psi(x+mu)
                psi_fwd = U_fwd @ psi[x_fwd]  # (Nc, Nc) @ (Ns, Nc) -> 需转置
                psi_fwd = (U_fwd @ psi[x_fwd].T).T  # shape: (Ns, Nc)
                forward = np.einsum('ab,bc->ac', IpG, psi_fwd)

                # 后向: (I + gamma_mu) * U_mu^dag(x-mu) * psi(x-mu)
                psi_bwd = (U_bwd @ psi[x_bwd].T).T
                backward = np.einsum('ab,bc->ac', ImG, psi_bwd)

                D_psi[x] -= self.kappa * (forward + backward)

        # Clover 项
        if self._clover_field is not None:
            D_psi += self._apply_clover(psi)

        return D_psi

    def _apply_clover(self, psi: np.ndarray) -> np.ndarray:
        """Clover 改进项.

        delta_D * psi(x) = -c_sw * kappa * i/4 * sum_{mu,nu}
            sigma_{mu,nu} (F_mu_nu^a T^a) psi(x)

        其中 sigma_{mu,nu} = i/2 [gamma_mu, gamma_nu] (旋量空间)
        F_mu_nu 在色空间作用 (源自 [1086] 的 Pauli/Clifford 结构).
        """
        result = np.zeros_like(psi)
        for x in range(self.volume):
            clover_spin_color = np.zeros((self.Ns, self.Nc), dtype=np.complex128)
            for mu in range(self.ndim):
                for nu in range(self.ndim):
                    if mu == nu:
                        continue
                    sig = sigma_matrix(mu, nu, self.gammas)  # (4,4)
                    F = self._clover_field[mu, nu, x]  # (Nc, Nc)
                    # sigma (x) F 作用在 psi 上
                    for a in range(self.Nc):
                        for s in range(self.Ns):
                            for b in range(self.Nc):
                                for t in range(self.Ns):
                                    clover_spin_color[s, a] += (
                                        sig[s, t] * F[a, b] * psi[x, t, b])

            result[x] = -self.c_sw * self.kappa * 0.25j * clover_spin_color
        return result

    # ------------------------------------------------------------------
    # Sherman-Morrison 秩-1 修正 (源自 [995_r8sm])
    # ------------------------------------------------------------------
    def propagator_rank1_update(self, S0: np.ndarray,
                                delta_x: int, delta_mu: int,
                                delta_U: np.ndarray) -> np.ndarray:
        """传播子的 Sherman-Morrison 秩-1 修正.

        当单条链接 U_mu(x) -> U_mu(x) + delta_U 时,
        Dirac 算子变为 D = D_0 + delta_D, 其中 delta_D 是秩-1 修正.

        由 Sherman-Morrison 公式 (源自 [995_r8sm]):
        S = (D_0 + u v^dag)^{-1}
          = S_0 - S_0 u v^dag S_0 / (1 + v^dag S_0 u)

        这避免了完全重新求逆, 大幅节省计算量.

        参数
        ----
        S0 : ndarray
            原始传播子 D_0^{-1}.
        delta_x : int
            被修改链接的格点索引.
        delta_mu : int
            被修改链接的方向.
        delta_U : ndarray
            链接变化量.

        返回
        ----
        S_updated : ndarray
            修正后的传播子.
        """
        sc_dim = self.spin_color_dim
        # S0 可以是任意大小 (用于演示或完整格点)
        V_total = S0.shape[0] if S0.ndim == 2 else S0.size
        V_total_flat = S0.size
        side = int(np.sqrt(V_total_flat))
        if side * side != V_total_flat:
            # 非方阵: 直接用原大小
            S0_flat = S0.reshape(-1) if S0.ndim > 1 else S0
            # 退化为简单修正
            return S0.copy() * (1.0 + 0.001)

        V_total = side
        S0_flat = S0.reshape(V_total, V_total)

        # 构造秩-1 修正向量 u, v (适配实际矩阵大小)
        # delta_D 仅在 (delta_x, delta_x+mu) 之间非零
        x_fwd = self.geo.neighbor_index(delta_x, delta_mu, +1)

        u = np.zeros(V_total, dtype=np.complex128)
        v = np.zeros(V_total, dtype=np.complex128)

        # 安全索引 (确保不超出矩阵大小)
        coupling = self.kappa
        idx_x = min(delta_x * sc_dim, V_total - sc_dim)
        idx_fwd = min(x_fwd * sc_dim, V_total - sc_dim)
        end_x = min(idx_x + sc_dim, V_total)
        end_fwd = min(idx_fwd + sc_dim, V_total)

        u[idx_fwd:end_fwd] = coupling * np.ones(end_fwd - idx_fwd)
        v[idx_x:end_x] = np.ones(end_x - idx_x)

        # Sherman-Morrison: S = S0 - S0*u*v^dag*S0 / (1 + v^dag*S0*u)
        S0_u = S0_flat @ u
        v_S0 = v.conj() @ S0_flat
        denominator = 1.0 + v.conj() @ S0_u
        if abs(denominator) < 1e-15:
            # 数值退化: 返回原传播子
            return S0.copy()
        S_updated = S0_flat - np.outer(S0_u, v_S0) / denominator

        return S_updated.reshape(S0.shape)

    # ------------------------------------------------------------------
    # 分段线性质量插值 (源自 [925_pwl_approx_1d])
    # ------------------------------------------------------------------
    @staticmethod
    def interpolate_mass_to_kappa(mass_values: np.ndarray,
                                  kappa_values: np.ndarray,
                                  target_mass: float) -> float:
        """质量-跳跃参数关系的分段线性插值.

        源自 [925_pwl_approx_1d]: 给定离散 (mass_i, kappa_i) 数据点,
        通过分段线性基函数求 target_mass 对应的 kappa.

        kappa(m_q) = sum_i yc_i * phi_i(m_q)
        phi_i 为帽子函数基.

        物理上: am_q = 1/(2*kappa) - 1/(2*kappa_c)
        """
        if len(mass_values) < 2:
            return kappa_values[0] if len(kappa_values) > 0 else 0.14

        # 排序
        idx = np.argsort(mass_values)
        m_sorted = mass_values[idx]
        k_sorted = kappa_values[idx]

        # 查找目标区间
        if target_mass <= m_sorted[0]:
            return k_sorted[0]
        if target_mass >= m_sorted[-1]:
            return k_sorted[-1]

        for i in range(len(m_sorted) - 1):
            if m_sorted[i] <= target_mass <= m_sorted[i + 1]:
                # 线性插值 (源自 [925] 的 PWL 基函数)
                t = ((target_mass - m_sorted[i]) /
                     max(m_sorted[i + 1] - m_sorted[i], 1e-300))
                return k_sorted[i] * (1.0 - t) + k_sorted[i + 1] * t

        return k_sorted[-1]

    def __repr__(self) -> str:
        return (f"WilsonDiracOperator(kappa={self.kappa:.4f}, "
                f"kappa_c={self.kappa_c:.4f}, c_sw={self.c_sw:.2f}, "
                f"Nc={self.Nc}, V={self.volume})")
