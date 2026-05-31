"""
initial_conditions.py
=====================
宇宙学初始条件生成模块

基于 Zeldovich 近似生成暗物质粒子的初始位移与速度，
融入 latin_edge（拉丁超立方采样）、hermite_rule（Gauss-Hermite 求积）、
statistics（alnorm 正态分布、离散采样）的核心算法，
构建符合 ΛCDM 功率谱的高斯随机场初始条件。

核心物理公式
------------
原初功率谱（带 transfer function 近似）:
    P(k) = A_s k^{n_s} T²(k)

    其中 A_s 为振幅，n_s 为谱指数，T(k) 为 transfer function。
    采用 Eisenstein & Hu (1998) 零重子无质量中微子近似:
        T(k) = ln(1 + 2.34 q) / (2.34 q) · [1 + 3.89 q + (16.1 q)² + (5.46 q)³ + (6.71 q)⁴]^{-1/4}
        q = k / (Ω_m h²) · (T_cmb / 2.7)²  (单位 h Mpc⁻¹)

Zeldovich 近似:
    粒子拉格朗日坐标 q 的欧拉坐标为:
        x(q, t) = q + D(t) · S(q)
    速度为:
        v(q, t) = a² f H D(t) · S(q)

    其中 S(q) = -∇Ψ(q) 为位移场，Ψ 满足泊松方程:
        ∇²Ψ(q) = -δ(q)

    增长率 f ≈ Ω_m(a)^{0.55} (Linder 近似)。

窗函数与质量分配:
    采用 Cloud-in-Cell (CIC) 或 Nearest-Grid-Point (NGP) 分配。
    粒子质量:
        m_p = ρ̄ L³ / N_p = Ω_m ρ_{crit,0} L³ / N_p

Gauss-Hermite 求积用于验证速度分布函数:
    一维速度分布为高斯分布:
        f(v) = (1/√(2π σ_v²)) exp(-v²/(2σ_v²))
    其各阶矩通过 Gauss-Hermite 求积精确计算:
        ⟨v^{2n}⟩ = ∫ v^{2n} f(v) dv ≈ Σ_i w_i x_i^{2n}
    其中节点 x_i 与权重 w_i 来自 Hermite 多项式 H_n(x) 的零点。
"""

import numpy as np
from typing import Tuple
from statistics import variance_from_power_spectrum, tophat_window


class TransferFunction:
    """
    Eisenstein & Hu (1998) 零重子物质 transfer function。
    """

    def __init__(self, cosmology):
        self.Omega_m = cosmology.Omega_m
        self.h = cosmology.h
        self.T_cmb = cosmology.T_cmb

    def __call__(self, k: np.ndarray) -> np.ndarray:
        """
        计算 transfer function T(k)。

        Parameters
        ----------
        k : np.ndarray
            波数，单位 h/Mpc

        Returns
        -------
        T : np.ndarray
            transfer function 值
        """
        k = np.asarray(k, dtype=float)
        out = np.ones_like(k)
        mask = k > 0.0
        k_m = k[mask]
        q = k_m / (self.Omega_m * self.h ** 2) * (self.T_cmb / 2.7) ** 2
        # 避免 q=0 的奇点
        q = np.clip(q, 1e-10, None)
        ln_term = np.log(1.0 + 2.34 * q) / (2.34 * q)
        poly_term = (
            1.0
            + 3.89 * q
            + (16.1 * q) ** 2
            + (5.46 * q) ** 3
            + (6.71 * q) ** 4
        ) ** (-0.25)
        out[mask] = ln_term * poly_term
        return out


class PowerSpectrum:
    """
    ΛCDM 物质功率谱 P(k)。
    """

    def __init__(self, cosmology, transfer_fn: TransferFunction = None):
        self.cosmo = cosmology
        self.transfer = transfer_fn or TransferFunction(cosmology)
        # 通过 sigma8 归一化振幅
        self.A_s = self._normalize_amplitude()

    def _normalize_amplitude(self) -> float:
        """
        利用 σ₈ 约束归一化功率谱振幅。

        σ₈² = (1/2π²) ∫ k² P(k) W²(kR) dk,  R = 8 h⁻¹ Mpc
        """
        R8 = 8.0  # h⁻¹ Mpc
        k_arr = np.logspace(-4, 2, 2000)
        T_arr = self.transfer(k_arr)
        P_unnorm = k_arr ** self.cosmo.ns * T_arr ** 2
        sigma2_unnorm = variance_from_power_spectrum(R8, k_arr, P_unnorm)
        if sigma2_unnorm <= 0.0:
            sigma2_unnorm = 1e-30
        A = self.cosmo.sigma8 ** 2 / sigma2_unnorm
        return A

    def __call__(self, k: np.ndarray) -> np.ndarray:
        """
        P(k) = A_s k^{n_s} T²(k)
        """
        k = np.asarray(k, dtype=float)
        T = self.transfer(k)
        return self.A_s * (k ** self.cosmo.ns) * (T ** 2)


def latin_edge_sample(dim_num: int, point_num: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    Latin edge 采样（融入 latin_edge 核心算法）。

    在每维空间中等概率放置 point_num 个点，且每行/列恰有一个点。
    坐标取值为 (0, 1, ..., point_num-1) / (point_num - 1)。

    在宇宙学中用于:
        - 宇宙学参数空间的均匀探索
        - 初始粒子位置的规则化网格扰动
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    if point_num == 1:
        return np.full((dim_num, 1), 0.5)
    x = np.zeros((dim_num, point_num))
    for i in range(dim_num):
        perm = rng.permutation(point_num)
        x[i, :] = perm / (point_num - 1.0)
    return x


def gauss_hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 Gauss-Hermite 求积节点与权重（融入 hermite_rule 核心思想）。

    通过构造对称三对角 Jacobi 矩阵并求其特征值获得节点:
        J_{i,i} = 0
        J_{i,i+1} = √(i/2)

    节点 x_i 为 J 的特征值，权重 w_i = √π · v_i[0]²，
    其中 v_i 为第 i 个归一化特征向量的第一个分量。

    积分公式:
        ∫_{-∞}^{+∞} exp(-x²) f(x) dx ≈ Σ_i w_i f(x_i)

    标准正态分布矩的计算:
        ∫_{-∞}^{+∞} (1/√(2π)) exp(-x²/2) x^{2m} dx
        = (1/√π) Σ_i w_i (√2 x_i)^{2m}
    """
    if n <= 0:
        raise ValueError("n 必须为正")
    # 构造 Jacobi 矩阵（权重 exp(-x²) 对应的 Hermite 多项式）
    diag = np.zeros(n)
    offdiag = np.sqrt(0.5 * np.arange(1, n))
    # 使用 numpy 的 eigh 求解对称三对角矩阵
    J = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    nodes = eigenvalues
    # 权重
    weights = np.sqrt(np.pi) * eigenvectors[0, :] ** 2
    return nodes, weights


def generate_zeldovich_displacement(
    N: int,
    L: float,
    power_spectrum: PowerSpectrum,
    D_growth: float = 1.0,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Zeldovich 近似生成三维位移场与速度场。

    算法步骤:
        1. 在傅里叶空间生成 δ_k ~ N(0, P(k)/V)
        2. 通过泊松方程求引力势的傅里叶分量:
               Ψ_k = δ_k / k²
        3. 位移场 S = -∇Ψ，在傅里叶空间:
               S_k = -i k δ_k / k²
        4. 逆 FFT 得到实空间位移
        5. 速度 v = a² f H D(t) S

    Parameters
    ----------
    N : int
        每维网格数
    L : float
        盒子边长（Mpc/h）
    power_spectrum : PowerSpectrum
        功率谱对象
    D_growth : float
        线性增长因子（已归一化）
    rng : np.random.Generator

    Returns
    -------
    pos : np.ndarray, shape (N³, 3)
        粒子欧拉坐标
    vel : np.ndarray, shape (N³, 3)
        粒子速度
    delta : np.ndarray, shape (N, N, N)
        实空间密度扰动场
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    # 波数网格
    k_vec = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky, kz = np.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k_mag = np.sqrt(k2)

    # 功率谱采样
    Pk = power_spectrum(k_mag.flatten()).reshape(k_mag.shape)
    # 体积因子: δ_k 的方差为 P(k)/V
    V = L ** 3
    amplitude = np.sqrt(Pk / V)

    # 生成复高斯随机场
    real_part = rng.standard_normal((N, N, N))
    imag_part = rng.standard_normal((N, N, N))
    delta_k = (real_part + 1j * imag_part) * amplitude

    # Hermitian 对称性 (利用 numpy 自动处理，这里手动保证)
    for i in range(N):
        for j in range(N):
            for kk in range(N):
                ii = (N - i) % N
                jj = (N - j) % N
                kk2 = (N - kk) % N
                if (ii, jj, kk2) > (i, j, kk):
                    delta_k[ii, jj, kk2] = delta_k[i, j, kk].conjugate()
    delta_k[0, 0, 0] = delta_k[0, 0, 0].real  # 零模为实数

    # 实空间密度场
    delta = np.fft.ifftn(delta_k).real * (N ** 3)

    # 引力势 Ψ_k = δ_k / k²
    Psi_k = np.zeros_like(delta_k)
    mask = k2 > 0.0
    Psi_k[mask] = delta_k[mask] / k2[mask]

    # 位移场 S_k = -i k Ψ_k = -i k δ_k / k²
    Sx_k = -1j * kx * Psi_k
    Sy_k = -1j * ky * Psi_k
    Sz_k = -1j * kz * Psi_k

    Sx = np.fft.ifftn(Sx_k).real * (N ** 3)
    Sy = np.fft.ifftn(Sy_k).real * (N ** 3)
    Sz = np.fft.ifftn(Sz_k).real * (N ** 3)

    # 拉格朗日网格坐标
    qx = np.linspace(0.0, L, N, endpoint=False)
    qgrid = np.meshgrid(qx, qx, qx, indexing="ij")

    # 应用 Zeldovich 位移
    x_pos = qgrid[0] + D_growth * Sx
    y_pos = qgrid[1] + D_growth * Sy
    z_pos = qgrid[2] + D_growth * Sz

    # 周期性边界条件
    x_pos = x_pos % L
    y_pos = y_pos % L
    z_pos = z_pos % L

    pos = np.stack([x_pos.ravel(), y_pos.ravel(), z_pos.ravel()], axis=1)

    # 速度场: v = a² f H D S (这里取 a=1, f≈1 的简化)
    f_growth = 1.0  # 近似值
    a_scale = 1.0
    # Hubble 参数单位转换因子
    H_a = 100.0  # km/s/Mpc 量级
    vel_factor = a_scale ** 2 * f_growth * H_a * D_growth  # 简化单位
    vel = vel_factor * np.stack([Sx.ravel(), Sy.ravel(), Sz.ravel()], axis=1)

    return pos, vel, delta


def particle_mass_from_cosmology(N: int, L: float, cosmology) -> float:
    """
    计算单个模拟粒子的质量:
        m_p = Ω_m ρ_{crit,0} L³ / N³
    """
    n_part = N ** 3
    return cosmology.Omega_m * cosmology.rho_crit_0 * (L ** 3) / n_part


if __name__ == "__main__":
    from cosmology import Cosmology

    cosmo = Cosmology()
    ps = PowerSpectrum(cosmo)
    k_test = np.logspace(-3, 1, 100)
    P_test = ps(k_test)
    print(f"P(k=0.1) = {np.interp(0.1, k_test, P_test):.4e}")

    # Latin edge 采样测试
    latin = latin_edge_sample(3, 8)
    print("Latin edge 采样 shape:", latin.shape)

    # Gauss-Hermite 节点测试
    nodes, weights = gauss_hermite_nodes_weights(8)
    # 验证 ∫ exp(-x²) dx = √π
    integral = np.sum(weights)
    print(f"Gauss-Hermite ∫ exp(-x²) dx = {integral:.8f} (理论 √π = {np.sqrt(np.pi):.8f})")

    # 生成初始条件（小规模测试）
    pos, vel, delta = generate_zeldovich_displacement(16, 100.0, ps, D_growth=1.0)
    print(f"初始位置 shape: {pos.shape}, 速度 shape: {vel.shape}")
    print(f"密度场 std: {delta.std():.4f}")
