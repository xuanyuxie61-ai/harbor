"""
high_order_fd.py
================

高阶有限差分算子, 用于格点费米子的离散化与 Symanzik 改进.

物理动机:
---------
连续 Dirac 作用量:
    S_F = ∫ d^4x ψ̄(x) (γ_μ D_μ + m) ψ(x)
其中协变导数 D_μ = ∂_μ - i g A_μ.

在格点上, 一阶导数的朴素离散化:
    ∇_μ^{(1)} f(x) = (f(x + a μ̂) - f(x - a μ̂)) / (2a)
具有 O(a^2) 误差.

为消除 discretization 误差, 采用 Symanzik 改进方案:

1. Naik 导数 (Follana et al. 2002, HISQ 改进):
    ∇_μ^{(Naik)} f(x) = (27/24) [f(x+μ̂) - f(x-μ̂)] / (2a)
                       - (1/24)  [f(x+3μ̂) - f(x-3μ̂)] / (6a)
   消除 O(a^2) 误差, 留下 O(a^4).

2. 高阶有限差分 (一般 N 阶):
    ∇_μ^{(N)} f(x) = Σ_{k=0}^{(N-1)/2} c_k [f(x+(2k+1)μ̂) - f(x-(2k+1)μ̂)] / (a (2k+1))
   其中 c_k 由 Taylor 展开匹配确定.

3. Clover (Sheikholeslami-Wohlert) 项:
    δS_SW = i c_SW a g / 4 ψ̄(x) σ_μν F_μν(x) ψ(x)
   其中 σ_μν = (i/2) [γ_μ, γ_ν], F_μν 为 clover 场强.

4. 动量空间色散关系:
    改进导数的色散关系 p̃_μ(p) 应更接近连续 p_μ:
        p̃_μ(p) = (1/a) Σ_k c_k sin((2k+1) a p_μ)
    在 Brillouin 区 |a p_μ| ≤ π 内最小化误差.

本模块融合种子项目:
  - 270_dfield9: 高阶 Runge-Kutta 积分思想
      推广到有限差分系数的设计: 通过匹配 Taylor 展开至 O(h^{2n})
  - 020_artery_pde: PDE 右侧的时间演化
      格点 Dirac 方程的时间演化 (用于 CG 求解器)
  - 1333_triangulation_boundary_nodes: 边界处理
      高阶梯度算子跨越边界时的相位修正
"""

import numpy as np
from typing import List, Tuple, Optional
from lattice_geometry import LatticeGeometry


# ============================================================
# 有限差分系数
# ============================================================

def fd_coefficients_central(order: int) -> List[float]:
    """计算中心差分的系数 c_k.

    对于 N 阶精度 (N 为偶数):
        f'(x) ≈ Σ_{k=0}^{N/2-1} c_k [f(x + (2k+1)h) - f(x - (2k+1)h)] / (2h)

    系数由 Taylor 展开匹配确定:
        Σ_k c_k (2k+1)^{2m-1} = δ_{m,1}  for m = 1, ..., N/2

    例:
        order=2: c = [1]  (标准中心差分)
        order=4: c = [4/3, -1/6]  (4 阶精度)
        order=6: c = [3/2, -3/20, 1/90]  (6 阶精度)
        order=8: c = [8/5, -4/15, 4/105, -1/280]  (8 阶)
    """
    if order == 2:
        return [1.0]
    elif order == 4:
        return [4.0/3.0, -1.0/6.0]
    elif order == 6:
        return [3.0/2.0, -3.0/20.0, 1.0/90.0]
    elif order == 8:
        return [8.0/5.0, -4.0/15.0, 4.0/105.0, -1.0/280.0]
    else:
        # 通用求解: 构建线性方程组
        n = order // 2
        A = np.zeros((n, n), dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)
        b[0] = 1.0
        for m in range(n):
            for k in range(n):
                A[m, k] = (2 * k + 1) ** (2 * m + 1)
        c = np.linalg.solve(A, b)
        return list(c)


def naik_coefficients() -> List[float]:
    """Naik 导数的系数 (HISQ 树级改进).

    Naik 导数是 3-link 改进的一阶导数:
        ∇^{Naik} = (27/24) ∇^{(1)} - (1/24) ∇^{(3)}

    其中 ∇^{(3)} f(x) = [f(x+3h) - f(x-3h)] / (6h).

    系数:
        c_0 = 9/8 (1-link 部分)
        c_1 = 1/24 (3-link 部分, 带负号)
    """
    return [9.0 / 8.0, -1.0 / 24.0]


def dispersion_relation(coeffs: List[float], p_times_a: np.ndarray) -> np.ndarray:
    """计算改进导数的动量空间色散关系.

    p̃(a p) = Σ_k c_k sin((2k+1) a p)

    连续极限: p̃ → a p 当 a p → 0.

    参数:
        coeffs: 有限差分系数 c_k
        p_times_a: 无量纲动量 a p ∈ [-π, π]
    """
    p_tilde = np.zeros_like(p_times_a)
    for k, c_k in enumerate(coeffs):
        p_tilde += c_k * np.sin((2 * k + 1) * p_times_a)
    return p_tilde


def dispersion_error(coeffs: List[float], n_points: int = 1000) -> float:
    """计算色散误差的 RMS:

    ε = √( (1/π) ∫_{-π}^{π} (p̃(p) - p)^2 dp )

    用于评估有限差分方案的精度.
    """
    p_vals = np.linspace(-np.pi, np.pi, n_points)
    p_tilde = dispersion_relation(coeffs, p_vals)
    err_sq = (p_tilde - p_vals) ** 2
    return float(np.sqrt(np.trapz(err_sq, p_vals) / (2 * np.pi)))


# ============================================================
# 格点导数算子
# ============================================================

class LatticeDerivative:
    """格点导数算子的统一接口.

    支持:
        - 朴素 1-link 导数 (O(a²))
        - Naik 3-link 导数 (O(a⁴))
        - 任意高阶中心差分
    """

    def __init__(self, geometry: LatticeGeometry,
                 order: int = 2,
                 use_naik: bool = False,
                 custom_coeffs: Optional[List[float]] = None):
        """
        参数:
            geometry: 格点几何
            order: 有限差分精度 (2, 4, 6, 8)
            use_naik: 是否使用 Naik 3-link 导数 (仅对 order=2 有效)
            custom_coeffs: 自定义系数 (覆盖 order)
        """
        self.geom = geometry
        self.order = order
        self.use_naik = use_naik

        if custom_coeffs is not None:
            self.coeffs = custom_coeffs
        elif use_naik:
            self.coeffs = naik_coefficients()
        else:
            self.coeffs = fd_coefficients_central(order)

        self.dispersion_err = dispersion_error(self.coeffs)

    def apply_scalar(self, field: np.ndarray, mu: int,
                     boundary_phase: float = 0.0) -> np.ndarray:
        """对标量场应用改进导数.

        ∇_μ f(x) = Σ_k c_k [f(x + (2k+1) μ̂) - f(x - (2k+1) μ̂)] / (2 (2k+1) a)

        参数:
            field: shape = (volume,) 的场值
            mu: 方向 (0, 1, 2, 3)
            boundary_phase: 跨越时间边界的相位 (费米子为 π, 规范场为 0)
        """
        grad = np.zeros_like(field)
        phase_factor = np.exp(1j * boundary_phase) if abs(boundary_phase) > 1e-10 else 1.0

        for k, c_k in enumerate(self.coeffs):
            hop = 2 * k + 1
            # 正向: field(x + hop * μ̂)
            field_plus = np.zeros_like(field)
            for idx in range(self.geom.volume):
                c = self.geom.idx_to_coord(idx)
                c_new = c.copy()
                L = self.geom.Ns if mu < 3 else self.geom.Nt
                # 处理跨越边界
                crosses_boundary = (c[mu] + hop >= L)
                c_new[mu] = (c[mu] + hop) % L
                idx_new = self.geom.coord_to_idx(*c_new)
                field_plus[idx] = field[idx_new]
                if crosses_boundary and mu == 3:
                    field_plus[idx] *= phase_factor

            # 反向: field(x - hop * μ̂)
            field_minus = np.zeros_like(field)
            for idx in range(self.geom.volume):
                c = self.geom.idx_to_coord(idx)
                c_new = c.copy()
                L = self.geom.Ns if mu < 3 else self.geom.Nt
                crosses_boundary = (c[mu] - hop < 0)
                c_new[mu] = (c[mu] - hop) % L
                idx_new = self.geom.coord_to_idx(*c_new)
                field_minus[idx] = field[idx_new]
                if crosses_boundary and mu == 3:
                    field_minus[idx] *= np.conj(phase_factor)

            # 中心差分: c_k × (f_+ - f_-) / (2 hop)
            grad += c_k * (field_plus - field_minus) / (2.0 * hop)

        return grad

    def covariant_derivative(self, gf, field: np.ndarray, mu: int,
                              spinor: bool = False) -> np.ndarray:
        """协变导数 ∇_μ^{cov} ψ(x).

        对旋量场:
            ∇_μ^{(1)} ψ(x) = (1 / (2a)) [U_μ(x) ψ(x + μ̂) - U_μ^†(x - μ̂) ψ(x - μ̂)]

        Naik 改进:
            ∇_μ^{Naik} ψ(x) = (9/8) ∇_μ^{(1)} ψ(x)
                             - (1/24) ∇_μ^{(3)} ψ(x)
        其中:
            ∇_μ^{(3)} ψ(x) = (1 / (6a)) [U_μ(x) U_μ(x+μ̂) U_μ(x+2μ̂) ψ(x+3μ̂)
                                       - U_μ^†(x-μ̂) U_μ^†(x-2μ̂) U_μ^†(x-3μ̂) ψ(x-3μ̂)]

        参数:
            gf: GaugeField
            field: shape = (volume, n_spin × 3) 的旋量场
            mu: 方向
        """
        if spinor:
            n_components = field.shape[1]
        else:
            n_components = 1
            field = field.reshape(-1, 1)

        grad = np.zeros_like(field)

        for k, c_k in enumerate(self.coeffs):
            hop = 2 * k + 1
            # 正向: Wilson line × ψ(x + hop μ̂)
            for idx in range(self.geom.volume):
                # 构建 Wilson line: U_μ(x) U_μ(x+μ̂) ... U_μ(x+(hop-1)μ̂)
                wilson_line = np.eye(3, dtype=np.complex128)
                current = idx
                for h in range(hop):
                    wilson_line = wilson_line @ gf.links[mu, current]
                    current = self.geom.neighbor_idx(current, mu, +1)
                idx_plus = current
                # 应用 Wilson line
                psi_plus = field[idx_plus].reshape(-1, 3) if n_components > 1 else field[idx_plus].reshape(1, 3)
                transported = (wilson_line @ psi_plus.T).T
                if spinor:
                    grad[idx] += c_k * transported.flatten() / (2.0 * hop)
                else:
                    # 对标量场, 取色迹
                    grad[idx, 0] += c_k * np.trace(transported).real / (2.0 * hop * 3.0)

            # 反向: Wilson line^† × ψ(x - hop μ̂)
            for idx in range(self.geom.volume):
                wilson_line_bwd = np.eye(3, dtype=np.complex128)
                current = idx
                for h in range(hop):
                    current = self.geom.neighbor_idx(current, mu, -1)
                    wilson_line_bwd = wilson_line_bwd @ gf.links[mu, current].conj().T
                idx_minus = current
                psi_minus = field[idx_minus].reshape(-1, 3) if n_components > 1 else field[idx_minus].reshape(1, 3)
                transported_bwd = (wilson_line_bwd @ psi_minus.T).T
                if spinor:
                    grad[idx] -= c_k * transported_bwd.flatten() / (2.0 * hop)
                else:
                    grad[idx, 0] -= c_k * np.trace(transported_bwd).real / (2.0 * hop * 3.0)

        if not spinor:
            return grad.flatten()
        return grad


# ============================================================
# 色散分析与稳定性
# ============================================================

class DispersionAnalyzer:
    """分析有限差分算子的动量空间行为.

    用途:
        1. 评估 discretization 误差
        2. 确定最优步长 (最小化色散误差)
        3. 分析格点费米子的 doublers 抑制
    """

    def __init__(self, max_order: int = 8):
        self.max_order = max_order
        self.methods = {}
        for order in range(2, max_order + 1, 2):
            coeffs = fd_coefficients_central(order)
            self.methods[f'central_{order}'] = {
                'coeffs': coeffs,
                'error': dispersion_error(coeffs)
            }
        self.methods['naik'] = {
            'coeffs': naik_coefficients(),
            'error': dispersion_error(naik_coefficients())
        }

    def compare_methods(self, n_points: int = 100) -> dict:
        """比较各方法的色散误差.

        返回字典: {method_name: {'coeffs': ..., 'error': ..., 'max_dev': ...}}
        """
        p_vals = np.linspace(0, np.pi, n_points)
        results = {}
        for name, data in self.methods.items():
            p_tilde = dispersion_relation(data['coeffs'], p_vals)
            max_dev = float(np.max(np.abs(p_tilde - p_vals)))
            results[name] = {
                'coeffs': data['coeffs'],
                'rms_error': data['error'],
                'max_deviation': max_dev,
            }
        return results

    def find_optimal_mixing(self, w: float) -> dict:
        """计算 Naik + 4 阶的混合方案的色散误差.

        混合方案:
            ∇^{mix} = w ∇^{Naik} + (1-w) ∇^{(4)}

        参数:
            w: 混合权重 ∈ [0, 1]
        """
        c_naik = naik_coefficients()
        c_4 = fd_coefficients_central(4)
        # 需要对齐长度
        n_max = max(len(c_naik), len(c_4))
        c_naik_pad = c_naik + [0] * (n_max - len(c_naik))
        c_4_pad = c_4 + [0] * (n_max - len(c_4))
        c_mix = [w * a + (1 - w) * b for a, b in zip(c_naik_pad, c_4_pad)]
        return {
            'coeffs': c_mix,
            'weight': w,
            'error': dispersion_error(c_mix)
        }


# ============================================================
# Clover 项 (SW 改进)
# ============================================================

def clover_term_coefficient(beta: float, c_sw: float = 1.0) -> float:
    """Clover 项系数:

    δS_SW = i c_SW a g / 4 ψ̄ σ_μν F_μν ψ

    树级: c_SW = 1
    1-loop 改进 (Lüscher et al. 1997):
        c_SW = 1 + 0.265 g^2 - 0.026 g^4 + ...
    其中 g^2 = 6/β.
    """
    if c_sw is not None and c_sw != 1.0:
        return c_sw
    g_sq = 6.0 / beta
    return 1.0 + 0.265 * g_sq - 0.026 * g_sq ** 2


def pauli_tensor(mu: int, nu: int) -> np.ndarray:
    """Pauli 张量 σ_μν = (i/2) [γ_μ, γ_ν].

    采用 Dirac 表示的 γ 矩阵 (Euclidean):
        γ_0 = ( 0  I )   γ_k = ( 0   -i σ_k )
              ( I  0 )         ( i σ_k  0   )

    返回 4×4 复数矩阵.
    """
    # Euclidean gamma matrices (chiral representation)
    gamma = np.zeros((4, 4, 4), dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    sigma1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sigmas = [sigma1, sigma2, sigma3]

    # γ_0
    gamma[0, :2, 2:] = I2
    gamma[0, 2:, :2] = I2
    # γ_k
    for k in range(3):
        gamma[k+1, :2, 2:] = -1j * sigmas[k]
        gamma[k+1, 2:, :2] = 1j * sigmas[k]

    # σ_μν = (i/2) [γ_μ, γ_ν]
    comm = gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu]
    return 0.5j * comm
