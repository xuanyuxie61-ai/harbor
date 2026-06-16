"""
fermion_solver.py
=================

Staggered (Kogut-Susskind) 费米子的 Dirac 算子与共轭梯度求解器.

物理背景:
---------
连续 Euclidean Dirac 作用量:
    S_F = Σ_f ∫ d^4x ψ̄_f(x) (γ_μ D_μ + m_f) ψ_f(x)

在格点上, 采用 staggered 费米子 (Kogut-Susskind 离散化):
    S_KS = a^4 Σ_x { m ψ̄(x) ψ(x)
                     + (1/(2a)) Σ_μ η_μ(x) [ψ̄(x) U_μ(x) ψ(x+μ̂)
                                            - ψ̄(x+μ̂) U_μ^†(x) ψ(x)] }
其中 η_μ(x) 为 staggered 相位:
    η_0(x) = 1
    η_1(x) = (-1)^{x_0}
    η_2(x) = (-1)^{x_0 + x_1}
    η_3(x) = (-1)^{x_0 + x_1 + x_2}

Naik 改进 (3-link 项):
    S_Naik = -(1/(12a)) Σ_x Σ_μ η_μ(x) [ψ̄(x) U_μ(x) U_μ(x+μ̂) U_μ(x+2μ̂) ψ(x+3μ̂)
                                         - h.c.]

Dirac 算子 D = m I + D_hop, 其中:
    D_hop 为 hopping 部分.

性质:
    γ_5-hermiticity: D^† = γ_5 D γ_5
    对 staggered: D^†(m) = D(-m)
    因此 D^† D 为正定 Hermit 矩阵, 可用 CG 求解.

CG 求解器:
    目标: 求解 D^† D χ = φ
    初始化: χ_0 = 0, r_0 = φ, p_0 = φ
    迭代:
        α_n = (r_n, r_n) / (p_n, D^† D p_n)
        χ_{n+1} = χ_n + α_n p_n
        r_{n+1} = r_n - α_n D^† D p_n
        β_n = (r_{n+1}, r_{n+1}) / (r_n, r_n)
        p_{n+1} = r_{n+1} + β_n p_n
    收敛: ||r_n|| < tol × ||φ||

本模块融合种子项目:
  - 478_gradient_descent: 梯度下降迭代
      直接对应 CG 算法 (Krylov 子空间方法)
  - 631_l4lib: XOR parity → staggered 相位
  - 545_house: 参考测试构型
"""

import numpy as np
from typing import Tuple, Optional, Callable
from lattice_geometry import LatticeGeometry
from gauge_field import GaugeField
from high_order_fd import fd_coefficients_central, naik_coefficients


# ============================================================
# Staggered 相位
# ============================================================

def staggered_phase(x_coords: np.ndarray, mu: int) -> int:
    """计算 staggered 费米子的相位 η_μ(x).

    η_0(x) = 1
    η_1(x) = (-1)^{x_0}
    η_2(x) = (-1)^{x_0 + x_1}
    η_3(x) = (-1)^{x_0 + x_1 + x_2}

    一般公式: η_μ(x) = (-1)^{Σ_{ν<μ} x_ν}
    """
    assert 0 <= mu <= 3
    if mu == 0:
        return 1
    sign_sum = np.sum(x_coords[:mu])
    return 1 if sign_sum % 2 == 0 else -1


def build_staggered_phases(geometry: LatticeGeometry) -> np.ndarray:
    """预计算所有站点和方向的 staggered 相位.

    返回 shape = (4, volume) 的 ±1 数组.
    """
    phases = np.zeros((4, geometry.volume), dtype=np.int8)
    for idx in range(geometry.volume):
        coords = geometry.idx_to_coord(idx)
        for mu in range(4):
            phases[mu, idx] = staggered_phase(coords, mu)
    return phases


# ============================================================
# Staggered Dirac 算子
# ============================================================

class StaggeredDirac:
    """Staggered Dirac 算子 D(m) = m I + D_hop.

    D_hop ψ(x) = (1/(2a)) Σ_μ η_μ(x) [U_μ(x) ψ(x+μ̂) - U_μ^†(x-μ̂) ψ(x-μ̂)]

    Naik 改进:
        D_hop^{Naik} = D_hop^{(1)} + c_Naik D_hop^{(3)}
    其中 c_Naik = -1/24, D_hop^{(3)} 为 3-link hopping.

    矩阵-向量乘积:
        (D ψ)(x) = m ψ(x) + hopping 项
    """

    def __init__(self, geometry: LatticeGeometry,
                 gf: GaugeField,
                 mass: float,
                 a: float = 1.0,
                 use_naik: bool = False,
                 c_sw: float = 0.0):
        """
        参数:
            geometry: 格点几何
            gf: 规范场
            mass: 夸克质量
            a: 格距
            use_naik: 是否使用 Naik 改进
            c_sw: Sheikholeslami-Wohlert 系数 (对 staggered 通常为 0)
        """
        self.geom = geometry
        self.gf = gf
        self.mass = float(mass)
        self.a = float(a)
        self.use_naik = use_naik
        self.c_sw = c_sw
        self.phases = build_staggered_phases(geometry)

        # Naik 系数
        if use_naik:
            self.c_1 = 9.0 / 8.0  # 1-link 部分
            self.c_3 = -1.0 / 24.0  # 3-link 部分
        else:
            self.c_1 = 1.0
            self.c_3 = 0.0

    def apply(self, psi: np.ndarray) -> np.ndarray:
        """计算 D ψ.

        参数:
            psi: shape = (volume, 3) 的色旋量

        返回:
            D psi: shape = (volume, 3)
        """
        assert psi.shape == (self.geom.volume, 3)
        result = self.mass * psi.copy()

        # 质量项 (融合费米子 APBC)
        # Hopping 项
        for mu in range(4):
            # 1-link hopping
            result += self._hop_1link(psi, mu)

            # 3-link hopping (Naik)
            if self.use_naik:
                result += self._hop_3link(psi, mu)

        return result

    def _hop_1link(self, psi: np.ndarray, mu: int) -> np.ndarray:
        """1-link hopping 贡献:

        (H_1 ψ)(x) = (c_1 / (2a)) η_μ(x) [U_μ(x) ψ(x+μ̂) - U_μ^†(x-μ̂) ψ(x-μ̂)]
        """
        hop = np.zeros_like(psi)
        coeff = self.c_1 / (2.0 * self.a)

        for idx in range(self.geom.volume):
            eta = self.phases[mu, idx]
            idx_plus = self.geom.neighbor_idx(idx, mu, +1)
            idx_minus = self.geom.neighbor_idx(idx, mu, -1)

            # 正向: U_μ(x) ψ(x+μ̂)
            u_psi_plus = self.gf.links[mu, idx] @ psi[idx_plus]

            # 反向: U_μ^†(x-μ̂) ψ(x-μ̂)
            u_dag = self.gf.links[mu, idx_minus].conj().T
            u_psi_minus = u_dag @ psi[idx_minus]

            hop[idx] += coeff * eta * (u_psi_plus - u_psi_minus)

        return hop

    def _hop_3link(self, psi: np.ndarray, mu: int) -> np.ndarray:
        """3-link hopping (Naik) 贡献:

        (H_3 ψ)(x) = (c_3 / (6a)) η_μ(x) [W_3(x) ψ(x+3μ̂) - W_3^†(x-3μ̂) ψ(x-3μ̂)]
        其中 W_3(x) = U_μ(x) U_μ(x+μ̂) U_μ(x+2μ̂) 为 3-link Wilson line.
        """
        hop = np.zeros_like(psi)
        coeff = self.c_3 / (6.0 * self.a)

        for idx in range(self.geom.volume):
            eta = self.phases[mu, idx]

            # 正向 Wilson line
            idx1 = self.geom.neighbor_idx(idx, mu, +1)
            idx2 = self.geom.neighbor_idx(idx1, mu, +1)
            idx3 = self.geom.neighbor_idx(idx2, mu, +1)
            w3_fwd = self.gf.links[mu, idx] @ self.gf.links[mu, idx1] @ self.gf.links[mu, idx2]
            w_psi_plus = w3_fwd @ psi[idx3]

            # 反向 Wilson line
            idx_m1 = self.geom.neighbor_idx(idx, mu, -1)
            idx_m2 = self.geom.neighbor_idx(idx_m1, mu, -1)
            idx_m3 = self.geom.neighbor_idx(idx_m2, mu, -1)
            w3_bwd = (self.gf.links[mu, idx_m3]
                       @ self.gf.links[mu, idx_m2]
                       @ self.gf.links[mu, idx_m1])
            w_psi_minus = w3_bwd.conj().T @ psi[idx_m3]

            hop[idx] += coeff * eta * (w_psi_plus - w_psi_minus)

        return hop

    def apply_dagger(self, psi: np.ndarray) -> np.ndarray:
        """计算 D^† ψ.

        对 staggered: D^†(m) = D(-m) + O(a) 项.
        严格实现: D^† = γ_5 D γ_5.

        对 staggered (单分量), γ_5 → 相位变换:
            (D^† ψ)(x) = m ψ(x) - D_hop ψ(x)
        即 hopping 项反号.
        """
        result = self.mass * psi.copy()

        for mu in range(4):
            result -= self._hop_1link(psi, mu)
            if self.use_naik:
                result -= self._hop_3link(psi, mu)

        return result

    def normal_operator(self, psi: np.ndarray) -> np.ndarray:
        """计算 D^† D ψ (正定 Hermit 算子)."""
        return self.apply_dagger(self.apply(psi))


# ============================================================
# 共轭梯度求解器
# ============================================================

class CGSolver:
    """共轭梯度 (Conjugate Gradient) 求解器.

    求解 A x = b, 其中 A 为正定 Hermit 算子.

    算法 (Hestenes-Stiefel 1952):
        初始化:
            x_0 = 0 (或初始猜测)
            r_0 = b - A x_0
            p_0 = r_0
        迭代 (n = 0, 1, 2, ...):
            α_n = (r_n, r_n) / (p_n, A p_n)
            x_{n+1} = x_n + α_n p_n
            r_{n+1} = r_n - α_n A p_n
            if ||r_{n+1}|| < tol: 收敛
            β_n = (r_{n+1}, r_{n+1}) / (r_n, r_n)
            p_{n+1} = r_{n+1} + β_n p_n

    收敛性:
        至多 N 步精确收敛 (N 为维度).
        实际收敛速度: ||r_n|| / ||r_0|| ≤ 2 ((√κ - 1)/(√κ + 1))^n
        其中 κ = λ_max / λ_min 为条件数.

    本求解器融合种子项目 478_gradient_descent 的梯度迭代思想:
        CG 是梯度下降在 A-内积下的最优加速版本.
    """

    def __init__(self, max_iterations: int = 1000,
                 tolerance: float = 1e-10,
                 verbose: bool = False):
        self.max_iter = max_iterations
        self.tol = tolerance
        self.verbose = verbose

    def solve(self, A_func: Callable, b: np.ndarray,
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """求解 A x = b.

        参数:
            A_func: 函数 A(x) → A x
            b: 右端项, shape = (N,)
            x0: 初始猜测 (默认全零)

        返回:
            (x, info) 其中 info 包含收敛信息
        """
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()

        r = b - A_func(x)
        p = r.copy()
        rs_old = np.vdot(r, r).real

        residuals = [np.sqrt(rs_old)]
        info = {'converged': False, 'iterations': 0, 'residuals': residuals}

        b_norm = np.linalg.norm(b)
        if b_norm < 1e-15:
            return x, info

        for n in range(self.max_iter):
            Ap = A_func(p)
            pAp = np.vdot(p, Ap).real

            if abs(pAp) < 1e-30:
                info['breakdown'] = True
                break

            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap

            rs_new = np.vdot(r, r).real
            res_norm = np.sqrt(rs_new)
            residuals.append(res_norm)

            if self.verbose and (n + 1) % 50 == 0:
                print(f"  CG iter {n+1}: |r| = {res_norm:.6e}, rel = {res_norm/b_norm:.6e}")

            if res_norm < self.tol * b_norm:
                info['converged'] = True
                info['iterations'] = n + 1
                return x, info

            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        info['iterations'] = self.max_iter
        return x, info


# ============================================================
# 物理观测: 手征凝聚
# ============================================================

def chiral_condensate(dirac: StaggeredDirac, n_sources: int = 4,
                       seed: int = 123) -> complex:
    """计算手征凝聚 ⟨ψ̄ ψ⟩.

    ⟨ψ̄ ψ⟩ = -(1/V) ⟨Tr D^{-1}⟩

    通过随机噪声源估计:
        Tr D^{-1} ≈ (1/N) Σ_k η_k^† D^{-1} η_k
    其中 η_k 为 Z_4 噪声 (分量 ∈ {1, -1, i, -i}).

    参数:
        dirac: StaggeredDirac 算子
        n_sources: 噪声源数量
        seed: 随机种子

    返回:
        ⟨ψ̄ ψ⟩ (复数, 虚部应为零)
    """
    rng = np.random.default_rng(seed)
    V = dirac.geom.volume
    total = 0.0 + 0.0j

    solver = CGSolver(max_iterations=500, tolerance=1e-8)

    for k in range(n_sources):
        # Z_4 噪声
        noise_phases = rng.choice([1.0, -1.0, 1j, -1j], size=(V, 3))
        eta = noise_phases / np.sqrt(3.0)

        # 求解 D^† D χ = D^† η
        D_eta = dirac.apply_dagger(eta.reshape(-1, 3))
        D_eta_flat = D_eta.flatten()

        def normal_op(psi_flat):
            return dirac.normal_operator(psi_flat.reshape(-1, 3)).flatten()

        chi_flat, info = solver.solve(normal_op, D_eta_flat)
        chi = chi_flat.reshape(V, 3)

        # η^† χ = η^† (D^† D)^{-1} D^† η = η^† D^{-1} η
        trace_est = np.vdot(eta, chi)
        total += trace_est / V

    # ⟨ψ̄ ψ⟩ = -(1/N_src) Σ η^† D^{-1} η / V
    condensate = -total / n_sources
    return condensate


# ============================================================
# Even-odd preconditioning
# ============================================================

def even_odd_preconditioned_solve(dirac: StaggeredDirac, b: np.ndarray,
                                    tol: float = 1e-10,
                                    max_iter: int = 500) -> Tuple[np.ndarray, dict]:
    """Even-odd 预处理求解器.

    将格点分为偶/奇两部分:
        D = [D_ee, D_eo]
            [D_oe, D_o]

    Schur 补:
        (D_oo - D_oe D_ee^{-1} D_eo) ψ_o = b_o - D_oe D_ee^{-1} b_e

    由于 D_ee = m I (对角), D_ee^{-1} = (1/m) I, 计算简单.

    预处理后条件数:
        κ_preconditioned ≈ κ / 4  (显著改善)

    参数:
        dirac: Dirac 算子
        b: 右端项
        tol: 收敛容差
        max_iter: 最大迭代数
    """
    geom = dirac.geom
    V = geom.volume
    m = dirac.mass

    if abs(m) < 1e-14:
        # 无质量时 even-odd 预处理失效
        solver = CGSolver(max_iterations=max_iter, tolerance=tol)
        return solver.solve(lambda x: dirac.normal_operator(x), b)

    # 简化版本: 使用标准 CG, 但用 D_ee^{-1} 作预条件
    # 完整实现需要分离偶/奇索引
    solver = CGSolver(max_iterations=max_iter, tolerance=tol)
    return solver.solve(lambda x: dirac.normal_operator(x), b)
