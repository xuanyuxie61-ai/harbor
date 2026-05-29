"""
lattice_gauge.py
================
格点几何、规范场构型、plaquette 作用量与 SU(2) 群元参数化。

原项目映射：
  - 1127_sphere_stereograph：球极投影用于 SU(2) 李群参数化
  - 670_levy_dragon_chaos：迭代函数系统（IFS）用于规范场热化噪声

物理背景
--------
在格点 QCD 中，SU(N_c) 规范群的基本自由度是规范联络（link variable）
U_μ(x) ∈ SU(3)。本模块针对 SU(2) 子群进行示范，其核心作用量为 Wilson plaquette action：

    S_G = β / N_c * Σ_{x,μ<ν} Re Tr[ 1 - P_{μν}(x) ]

其中 plaquette 定义为

    P_{μν}(x) = U_μ(x) U_ν(x+μ̂) U_μ†(x+ν̂) U_ν†(x)

SU(2) 群元可参数化为四元数：

    U = a_0 I + i Σ_{k=1}^{3} a_k σ_k ,   Σ a_i^2 = 1

利用球极投影（stereographic projection），可将单位球面 S^3 上的 SU(2)
群元映射到 R^3，便于数值微分与优化。
"""

import numpy as np


class Lattice:
    """四维周期性格点。"""

    def __init__(self, nx: int = 4, ny: int = 4, nz: int = 4, nt: int = 8):
        self.dims = np.array([nx, ny, nz, nt], dtype=int)
        self.nd = 4
        self.vol = nx * ny * nz * nt
        self.shape = (nx, ny, nz, nt)

    def site_index(self, x: np.ndarray) -> int:
        """将四维坐标展平为一维索引（带周期边界）。"""
        x = np.mod(x, self.dims)
        return int(x[0] + self.dims[0] * (x[1] + self.dims[1] * (x[2] + self.dims[2] * x[3])))

    def index_to_site(self, idx: int) -> np.ndarray:
        """一维索引转四维坐标。"""
        x = np.zeros(4, dtype=int)
        x[0] = idx % self.dims[0]
        idx //= self.dims[0]
        x[1] = idx % self.dims[1]
        idx //= self.dims[1]
        x[2] = idx % self.dims[2]
        idx //= self.dims[2]
        x[3] = idx
        return x

    def neighbor(self, x: np.ndarray, mu: int, sign: int = 1) -> np.ndarray:
        """沿方向 mu (0..3) 移动 ±1 步，周期性边界。"""
        y = x.copy()
        y[mu] = (y[mu] + sign) % self.dims[mu]
        return y


def su2_random() -> np.ndarray:
    """
    生成随机 SU(2) 群元（Haar 测度）。

    算法：在单位球面 S^3 上均匀采样四元数 (a0, a1, a2, a3)。
    """
    v = np.random.randn(4)
    v /= np.linalg.norm(v)
    return np.array([[v[0] + 1j * v[3], v[1] + 1j * v[2]],
                     [-v[1] + 1j * v[2], v[0] - 1j * v[3]]])


def su2_identity() -> np.ndarray:
    return np.eye(2, dtype=complex)


def su2_dagger(u: np.ndarray) -> np.ndarray:
    return u.conj().T


def su2_trace(u: np.ndarray) -> complex:
    return np.trace(u)


def su2_stereographic_project(u: np.ndarray) -> np.ndarray:
    """
    对 SU(2) 群元进行广义球极投影到 R^3。

    参数化：U = a0 I + i a·σ，其中 a0^2 + |a|^2 = 1。
    以南极 S = (0,0,0,-1) 为投影中心，将 (a0, a1, a2, a3) 投影到 R^3：

        q_i = a_i / (1 + a0) ,   i = 1,2,3

    逆映射（用于从 R^3 重建 SU(2)）：

        a0 = (1 - |q|^2) / (1 + |q|^2)
        a_i = 2 q_i / (1 + |q|^2)

    Parameters
    ----------
    u : np.ndarray
        2x2 SU(2) 矩阵。

    Returns
    -------
    q : np.ndarray
        R^3 中的投影点。
    """
    a0 = u[0, 0].real
    a1 = u[0, 1].real
    a2 = u[0, 1].imag
    a3 = u[0, 0].imag
    eps = 1e-12
    denom = 1.0 + a0
    if abs(denom) < eps:
        denom = eps * np.sign(denom + eps)
    q = np.array([a1, a2, a3]) / denom
    return q


def su2_stereographic_inverse(q: np.ndarray) -> np.ndarray:
    """从 R^3 逆球极投影重建 SU(2) 群元。"""
    norm_sq = np.dot(q, q)
    denom = 1.0 + norm_sq
    a0 = (1.0 - norm_sq) / denom
    a1 = 2.0 * q[0] / denom
    a2 = 2.0 * q[1] / denom
    a3 = 2.0 * q[2] / denom
    return np.array([[a0 + 1j * a3, a1 + 1j * a2],
                     [-a1 + 1j * a2, a0 - 1j * a3]], dtype=complex)


class GaugeConfig:
    """
    SU(2) 规范场构型。

    数据结构：U[mu][x,y,z,t] 为 2x2 复矩阵。
    """

    def __init__(self, lattice: Lattice):
        self.lat = lattice
        self.U = np.zeros((4, *lattice.shape, 2, 2), dtype=complex)
        for mu in range(4):
            for idx in range(lattice.vol):
                x = lattice.index_to_site(idx)
                self.U[(mu, *x)] = su2_identity()

    def randomize(self):
        """将所有 link 随机化为 Haar 分布。"""
        for mu in range(4):
            for idx in range(self.lat.vol):
                x = self.lat.index_to_site(idx)
                self.U[(mu, *x)] = su2_random()

    def get_link(self, mu: int, x: np.ndarray) -> np.ndarray:
        x = np.mod(x, self.lat.dims)
        return self.U[(mu, *x)].copy()

    def set_link(self, mu: int, x: np.ndarray, u: np.ndarray):
        x = np.mod(x, self.lat.dims)
        self.U[(mu, *x)] = u.copy()

    def _link_array(self, mu: int) -> np.ndarray:
        """返回方向 mu 的链路数组视图 (nx,ny,nz,nt,2,2)。"""
        return self.U[mu]

    def plaquette(self, x: np.ndarray, mu: int, nu: int) -> np.ndarray:
        """
        计算 plaquette P_{μν}(x)。

        P_{μν}(x) = U_μ(x) U_ν(x+μ̂) U_μ†(x+ν̂) U_ν†(x)
        """
        u1 = self.get_link(mu, x)
        u2 = self.get_link(nu, self.lat.neighbor(x, mu, 1))
        u3 = su2_dagger(self.get_link(mu, self.lat.neighbor(x, nu, 1)))
        u4 = su2_dagger(self.get_link(nu, x))
        return u1 @ u2 @ u3 @ u4

    def average_plaquette(self) -> float:
        """计算所有 plaquette 的实部迹的平均值。"""
        s = 0.0
        count = 0
        nx, ny, nz, nt = self.lat.shape
        # 使用向量化加速：预取链路数组
        for mu in range(4):
            for nu in range(mu + 1, 4):
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            for it in range(nt):
                                x = np.array([ix, iy, iz, it])
                                p = self.plaquette(x, mu, nu)
                                s += su2_trace(p).real
                                count += 1
        return s / count if count > 0 else 0.0

    def wilson_action(self, beta: float = 2.4) -> float:
        """
        Wilson 纯规范作用量。

        S_G = β * Σ_{x,μ<ν} ( 1 - 1/N_c Re Tr P_{μν}(x) )
        对于 SU(2)，N_c = 2。
        """
        avg = self.average_plaquette()
        return beta * (1.0 - 0.5 * avg) * self.lat.vol * 6.0


def ifs_thermalize_gauge(gauge: GaugeConfig, n_iter: int = 50):
    """
    利用迭代函数系统（IFS）对规范场构型进行伪热化噪声注入。

    原项目映射：670_levy_dragon_chaos 的仿射迭代思想。

    算法：在每个格点上，以概率选择两种 SU(2) 更新映射之一：
        - 映射 0：U → V0 U，其中 V0 为随机群元（模拟热浴）
        - 映射 1：U → V1 U，其中 V1 为微小随机扰动（模拟微正则更新）
    通过多轮迭代，使规范场构型达到“混沌”均匀分布。

    Parameters
    ----------
    gauge : GaugeConfig
        规范场构型。
    n_iter : int
        IFS 迭代次数。
    """
    lat = gauge.lat
    for _ in range(n_iter):
        for idx in range(lat.vol):
            x = lat.index_to_site(idx)
            for mu in range(4):
                u = gauge.get_link(mu, x)
                if np.random.rand() < 0.5:
                    v = su2_random()
                else:
                    phi = np.random.randn(3) * 0.1
                    v = su2_stereographic_inverse(phi)
                gauge.set_link(mu, x, v @ u)
