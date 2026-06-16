"""
nonhermitian_hamiltonian.py
===========================
构造非厄米 SSH (Su-Schrieffer-Heeger) 模型与 Hatano-Nelson 模型的哈密顿量.
融合种子项目:
  - 481_graph_adj: 紧束缚晶格的邻接矩阵表示
  - 099_blend: 参数空间混合插值构造平滑哈密顿量族
  - 893_polynomial: 特征多项式代数 (判别式分析定位 EP)

物理模型:
  H_SSH(k) = [ gamma       t1 + t2*exp(-ik) ]
             [ t1 + t2*exp(ik)   -gamma     ]
  其中 t1, t2 为交错 hopping, gamma 为非厄米在位势.
  例外点 (EP) 出现在 det(H - E*I) = 0 且 dE/dk = 0 同时成立处.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, List, Optional


# ---------------------------------------------------------------------------
# 图邻接 → 紧束缚哈密顿量 (源自 481_graph_adj)
# ---------------------------------------------------------------------------
def adjacency_to_tight_binding(adj: np.ndarray,
                               on_site: Optional[np.ndarray] = None,
                               hopping_phase: complex = 1.0 + 0.0j) -> np.ndarray:
    """将无向/有向图邻接矩阵转为紧束缚哈密顿量.

    对于非厄米 Hatano-Nelson 模型, 有向边给出不对称 hopping:
        H_{ij} = t * exp(i*phi)  if adj[i,j] = 1
    在位能由 on_site 给出 (可为复数, 实现非厄米在位损耗/增益).

    数学表达:
        H = sum_i epsilon_i |i><i| + sum_{i,j} t_{ij} |i><j|
    其中 t_{ij} = adj[i,j] * hopping_phase.

    Parameters
    ----------
    adj : (N, N) 邻接矩阵 (0/1 或权重).
    on_site : (N,) 在位能; 默认纯虚数非厄米分布 i*gamma*(-1)^i.
    hopping_phase : 非厄米 hopping 相位因子.

    Returns
    -------
    H : (N, N) 复数哈密顿量矩阵.
    """
    n = adj.shape[0]
    if adj.shape != (n, n):
        raise ValueError("adj must be square")
    if on_site is None:
        gamma_nh = 0.3
        on_site = np.array([1j * gamma_nh * ((-1) ** i) for i in range(n)],
                           dtype=complex)
    if on_site.shape != (n,):
        raise ValueError("on_site shape mismatch")
    H = np.diag(on_site.astype(complex))
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] != 0:
                H[i, j] += adj[i, j] * hopping_phase
    return H


def ssh_hamiltonian_1d(t1: float, t2: float, gamma: float,
                       k: float) -> np.ndarray:
    """2-band 非厄米 SSH 模型的 2x2 哈密顿量.

    H_SSH(k) = [ i*gamma    t1 + t2*exp(-ik) ]
               [ t1 + t2*exp(ik)   -i*gamma  ]

    本征值:
        E_{\pm}(k) = \pm sqrt[(t1 + t2 cos k)^2 + (t2 sin k)^2 - gamma^2]
    EP 出现在 E_+ = E_- = 0, 即
        (t1 + t2 cos k_EP)^2 + (t2 sin k_EP)^2 = gamma^2
    """
    phase = np.exp(1j * k)
    hop = t1 + t2 * np.conj(phase)
    H = np.array([[1j * gamma, hop],
                  [t1 + t2 * phase, -1j * gamma]], dtype=complex)
    return H


def hatano_nelson_hamiltonian(N: int, t: float, g: float,
                              boundary: str = "periodic") -> np.ndarray:
    """Hatano-Nelson 模型: 非厄米非对称 hopping 的 1D 晶格.

    H = sum_i [ t*exp(+g) |i><i+1| + t*exp(-g) |i+1><i| ]

    参数 g 控制非厄米强度; g=0 回到厄米情况.
    边界条件: periodic / open.

    Returns
    -------
    H : (N, N) 复数矩阵.
    """
    if N < 3:
        raise ValueError("Hatano-Nelson requires N >= 3")
    t_plus = t * np.exp(+g)
    t_minus = t * np.exp(-g)
    H = np.zeros((N, N), dtype=complex)
    for i in range(N - 1):
        H[i, i + 1] = t_plus
        H[i + 1, i] = t_minus
    if boundary == "periodic":
        H[0, N - 1] = t_minus
        H[N - 1, 0] = t_plus
    elif boundary != "open":
        raise ValueError("boundary must be 'periodic' or 'open'")
    return H


# ---------------------------------------------------------------------------
# 混合插值构造哈密顿量族 (源自 099_blend)
# ---------------------------------------------------------------------------
def blend_hamiltonian_101(r: float, H0: np.ndarray,
                          H1: np.ndarray) -> np.ndarray:
    """线性混合: H(r) = (1-r)*H0 + r*H1.

    用于在参数空间构造连接两个哈密顿量的路径, 追踪沿路径的 EP.
    源自 blend_101 的端点插值思想.
    """
    if H0.shape != H1.shape:
        raise ValueError("H0 and H1 must have same shape")
    if not (0.0 <= r <= 1.0):
        raise ValueError("blend parameter r must be in [0, 1]")
    return (1.0 - r) * H0 + r * H1


def blend_hamiltonian_123(r: float, s: float, t_param: float,
                          Hcorners: Dict[str, np.ndarray]) -> np.ndarray:
    """三参数混合: 在 (r,s,t) 参数立方体内插值 8 个顶点哈密顿量.

    源自 blend_123 的三线性混合, 用于扫描三维参数空间 (t1, t2, gamma).
    Hcorners 需包含键 '000', '001', ..., '111'.

    H(r,s,t) = sum_{a,b,c in {0,1}} B_{abc}(r,s,t) * H_{abc}
    其中 B_{abc} 为三线性基函数.
    """
    required = ['000', '001', '010', '011', '100', '101', '110', '111']
    for key in required:
        if key not in Hcorners:
            raise ValueError(f"missing corner {key}")
    ref_shape = Hcorners['000'].shape
    result = np.zeros(ref_shape, dtype=complex)
    for ia, a in enumerate([1.0 - r, r]):
        for ib, b in enumerate([1.0 - s, s]):
            for ic, c in enumerate([1.0 - t_param, t_param]):
                key = f"{ia}{ib}{ic}"
                result += a * b * c * Hcorners[key]
    return result


# ---------------------------------------------------------------------------
# 特征多项式与判别式 (源自 893_polynomial)
# ---------------------------------------------------------------------------
def characteristic_polynomial_2x2(H: np.ndarray) -> Tuple[complex, complex, complex]:
    """计算 2x2 矩阵的特征多项式系数: lambda^2 + c1*lambda + c0 = 0.

    c1 = -tr(H), c0 = det(H).
    判别式 Delta = c1^2 - 4*c0.
    EP 对应 Delta = 0.
    """
    if H.shape != (2, 2):
        raise ValueError("expected 2x2 matrix")
    c1 = -np.trace(H)
    c0 = np.linalg.det(H)
    return c1, c0


def discriminant_ep_indicator(H: np.ndarray) -> complex:
    """判别式 Delta = tr(H)^2 - 4*det(H).

    |Delta| -> 0 指示例外点临近.
    """
    c1, c0 = characteristic_polynomial_2x2(H)
    return c1 ** 2 - 4.0 * c0


def polynomial_multiply_1d(p1: np.ndarray,
                           p2: np.ndarray) -> np.ndarray:
    """一维多项式乘法 (源自 polynomial_mul).

    p1, p2 为按升幂排列的系数数组.
    用于构造高阶特征多项式.
    """
    n1, n2 = len(p1), len(p2)
    if n1 == 0 or n2 == 0:
        return np.array([], dtype=complex)
    result = np.zeros(n1 + n2 - 1, dtype=complex)
    for i in range(n1):
        for j in range(n2):
            result[i + j] += p1[i] * p2[j]
    return result


def polynomial_differentiate_1d(p: np.ndarray,
                                order: int = 1) -> np.ndarray:
    """一维多项式求导 (源自 polynomial_dif).

    用于计算 d Delta / dk, 辅助 EP 定位.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    coef = np.array(p, dtype=complex)
    for _ in range(order):
        if len(coef) < 2:
            return np.array([0.0 + 0.0j])
        coef = np.array([(i + 1) * coef[i + 1] for i in range(len(coef) - 1)],
                        dtype=complex)
    return coef


def build_spectral_phase_diagram(t1_range: np.ndarray,
                                 t2_range: np.ndarray,
                                 gamma: float,
                                 k_samples: int = 64) -> np.ndarray:
    """扫描 (t1, t2) 参数空间, 计算最小 |Delta(k)| 标记 EP 区域.

    Returns
    -------
    min_delta : (len(t1_range), len(t2_range)) 数组, 每点最小 |Delta|.
    """
    n1, n2 = len(t1_range), len(t2_range)
    min_delta = np.zeros((n1, n2))
    k_vals = np.linspace(0, 2 * np.pi, k_samples, endpoint=False)
    for i, t1 in enumerate(t1_range):
        for j, t2 in enumerate(t2_range):
            d_min = np.inf
            for k in k_vals:
                H = ssh_hamiltonian_1d(t1, t2, gamma, k)
                d = abs(discriminant_ep_indicator(H))
                if d < d_min:
                    d_min = d
            min_delta[i, j] = d_min
    return min_delta
