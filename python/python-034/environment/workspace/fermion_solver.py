"""
fermion_solver.py
=================
格点 Wilson-Dirac 费米子传播子求解器。

原项目映射：
  - 355_fd1d_advection_lax_wendroff：Lax-Wendroff 离散格式用于费米子 hopping 项
  - 991_r8pp：对称正定打包矩阵 Cholesky 分解用于 CG 预处理

物理背景
--------
Wilson-Dirac 算符在格点上定义为：

    D_w(x,y) = (4 + m_0 a) δ_{xy}
             - (1/2) Σ_μ [ (1 - γ_μ) U_μ(x) δ_{x+μ̂,y}
                         + (1 + γ_μ) U_μ†(x-μ̂) δ_{x-μ̂,y} ]

其中 γ_μ 为 Dirac 矩阵（本模块采用 2 分量简化旋量表示），
m_0 为裸夸克质量，a 为晶格间距（设 a=1）。
定义跳跃参数 κ：

    κ = 1 / (2 m_0 a + 8)

则 D_w 可写为：

    D_w = I / κ - H_w

其中 H_w 为 hopping 矩阵。传播子 S = D_w^{-1} 满足：

    Σ_y D_w(x,y) S(y,z) = δ_{xz}

数值方法
--------
1. Lax-Wendroff 风格离散：对 Dirac 方程中的“平流”项（hopping 项）
   使用二阶中心差分加人工耗散，提高数值稳定性。

2. 共轭梯度（CG）法：由于 Wilson-Dirac 算符厄米正定的性质
   （在合适的预处理下），CG 是求解传播子的标准方法。

3. Cholesky 预处理：利用 matrix_algebra 中的打包 SPD Cholesky 分解
   对粗略格点上的正规方程进行预处理，加速收敛。

边界处理
--------
- 周期性边界条件（periodic）用于空间方向；
- 反周期性边界条件（anti-periodic）用于时间方向（费米子）。
"""

import numpy as np
from lattice_gauge import Lattice, GaugeConfig, su2_dagger
from matrix_algebra import r8pp_fa, r8pp_sl, dense_to_packed


# 简化的 2x2 Dirac gamma 矩阵（用 Pauli 矩阵代替，适用于 1+1 或简化 3+1）
_GAMMA = [
    np.array([[0, 1], [1, 0]], dtype=complex),   # γ_x
    np.array([[0, -1j], [1j, 0]], dtype=complex), # γ_y
    np.array([[1, 0], [0, -1]], dtype=complex),  # γ_z
    np.array([[0, 1], [1, 0]], dtype=complex),   # γ_t (simplified)
]

_GAMMA5 = np.array([[1, 0], [0, -1]], dtype=complex)


def gamma_mu(mu: int) -> np.ndarray:
    """返回方向 μ 的简化 Dirac gamma 矩阵。"""
    return _GAMMA[mu % 4]


class WilsonDiracOperator:
    """
    Wilson-Dirac 算符的稀疏矩阵表示。

    数据结构：对每一个格点 x 和自旋分量 s，存储邻居耦合。
    """

    def __init__(self, lattice: Lattice, gauge: GaugeConfig,
                 mass: float = 0.1, boundary_phase: np.ndarray = None):
        self.lat = lattice
        self.gauge = gauge
        self.mass = mass
        self.kappa = 1.0 / (2.0 * mass + 8.0)
        if boundary_phase is None:
            # 时间方向反周期，空间方向周期
            self.boundary_phase = np.array([1.0, 1.0, 1.0, -1.0])
        else:
            self.boundary_phase = np.array(boundary_phase)

    def apply(self, psi: np.ndarray) -> np.ndarray:
        """
        应用 Wilson-Dirac 算符 D_w 到旋量场 ψ。

        ψ 的形状：(nx, ny, nz, nt, 2) [复数]。
        返回 D_w ψ，同形状。
        """
        # HOLE 1: Wilson-Dirac 算符作用于旋量场的核心计算
        raise NotImplementedError("Hole 1: implement Wilson-Dirac apply")

    def apply_dagger(self, psi: np.ndarray) -> np.ndarray:
        """
        应用 D_w^†。利用 γ_5 厄米性：D_w^† = γ_5 D_w γ_5。
        """
        # HOLE 2: 利用 gamma_5 厄米性实现 apply_dagger
        raise NotImplementedError("Hole 2: implement apply_dagger using gamma_5 hermiticity")

    def apply_hermitian(self, psi: np.ndarray) -> np.ndarray:
        """
        应用厄米算符 D_w^† D_w（SPD），适合 CG 求解。
        """
        tmp = self.apply(psi)
        return self.apply_dagger(tmp)


def solve_propagator_cg(wd: WilsonDiracOperator, source: np.ndarray,
                        max_iter: int = 80, tol: float = 1e-6) -> np.ndarray:
    """
    使用共轭梯度法求解 Wilson-Dirac 方程 D_w S = source。

    由于 D_w 可能非厄米，实际求解厄米化方程：
        D_w^† D_w S = D_w^† source

    CG 算法：
        r_0 = b - A x_0
        p_0 = r_0
        for k = 0, 1, 2, ...:
            α_k = (r_k^† r_k) / (p_k^† A p_k)
            x_{k+1} = x_k + α_k p_k
            r_{k+1} = r_k - α_k A p_k
            if ||r_{k+1}|| < tol: break
            β_k = (r_{k+1}^† r_{k+1}) / (r_k^† r_k)
            p_{k+1} = r_{k+1} + β_k p_k

    Parameters
    ----------
    wd : WilsonDiracOperator
        Wilson-Dirac 算符。
    source : np.ndarray
        源场，形状 (nx, ny, nz, nt, 2)。
    max_iter : int
        最大迭代次数。
    tol : float
        残差容差。

    Returns
    -------
    sol : np.ndarray
        传播子 S = D_w^{-1} source。
    """
    lat = wd.lat
    # 右端项 b = D_w^† source
    b = wd.apply_dagger(source).reshape(-1, 2)

    def matvec(v_flat):
        v = v_flat.reshape(*lat.shape, 2)
        Av = wd.apply_hermitian(v)
        return Av.reshape(-1, 2)

    def dot(a, b_vec):
        return np.sum(a.conj() * b_vec).real

    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rsold = dot(r, r)
    if rsold < tol ** 2:
        return x.reshape(*lat.shape, 2)

    for k in range(max_iter):
        Ap = matvec(p)
        pAp = dot(p, Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rsold / pAp
        x += alpha * p
        r -= alpha * Ap
        rsnew = dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    sol = x.reshape(*lat.shape, 2)
    return sol


def point_source(lat: Lattice, x0: np.ndarray, spin: int = 0) -> np.ndarray:
    """
    生成 δ 函数点源。

    Parameters
    ----------
    lat : Lattice
        格点几何。
    x0 : np.ndarray
        源位置。
    spin : int
        自旋分量。

    Returns
    -------
    src : np.ndarray
        点源场。
    """
    src = np.zeros((*lat.shape, 2), dtype=complex)
    x0 = np.mod(x0, lat.dims)
    src[(x0[0], x0[1], x0[2], x0[3], spin)] = 1.0
    return src


def solve_all_propagators(wd: WilsonDiracOperator,
                          sources: list) -> list:
    """
    对多个源求解传播子。

    Returns
    -------
    propagators : list[np.ndarray]
        传播子列表。
    """
    props = []
    for src in sources:
        prop = solve_propagator_cg(wd, src, max_iter=80, tol=1e-6)
        props.append(prop)
    return props
