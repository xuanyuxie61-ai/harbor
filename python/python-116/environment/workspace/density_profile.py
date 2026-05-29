"""
density_profile.py
跨膜密度轮廓的谱表示模块

本模块使用 Bernstein 多项式基对脂质双分子层的跨膜质量密度
轮廓 ρ(z) 进行平滑表示与逼近。Bernstein 基具有良好的端点插值性质
和几何直观性，适合描述具有明确边界的膜系统。

参考种子项目: 078_bernstein_polynomial (Bernstein 多项式基函数)

物理模型:
    跨膜质量密度轮廓 ρ(z) 满足:
        ρ(z) = ρ_w(z) + ρ_head(z) + ρ_tail(z) + ρ_chol(z)
    其中 z 为垂直于膜平面的坐标，z=0 为膜中心。

    采用无量纲坐标 u = (z - z_min)/(z_max - z_min) ∈ [0,1]，
    则 Bernstein 逼近:
        ρ̃(u) = Σ_{k=0}^{n} b_k B_{n,k}(u)

    Bernstein 基函数:
        B_{n,k}(u) = C(n,k) * u^k * (1-u)^{n-k}

    该基具有保形性（variation-diminishing），且端点插值:
        ρ̃(0) = b_0,  ρ̃(1) = b_n
"""

import numpy as np
from scipy.special import comb


def bernstein_basis(n, u):
    """
    计算 n 次 Bernstein 基函数 B_{n,k}(u), k=0..n。

    Parameters
    ----------
    n : int
        多项式次数，n ≥ 0。
    u : float or ndarray
        参数 u ∈ [0,1]。

    Returns
    -------
    B : ndarray, shape (..., n+1)
        B[..., k] = B_{n,k}(u)。
    """
    if n < 0:
        raise ValueError("n 必须非负。")
    u = np.atleast_1d(u)
    u = np.clip(u, 0.0, 1.0)

    if n == 0:
        return np.ones_like(u)

    B = np.zeros(u.shape + (n + 1,))
    # 递推计算避免直接求组合数的数值溢出
    B[..., 0] = 1.0 - u
    B[..., 1] = u

    for i in range(2, n + 1):
        B[..., i] = u * B[..., i - 1]
        for j in range(i - 1, 0, -1):
            B[..., j] = u * B[..., j - 1] + (1.0 - u) * B[..., j]
        B[..., 0] = (1.0 - u) * B[..., 0]

    return B


def bernstein_to_monomial_matrix(n):
    """
    Bernstein 基到单项式基的转换矩阵 M，满足:
        x^k = Σ_j M[k,j] B_{n,j}(x)
    或等价地，系数转换 b_bernstein = M^T @ b_monomial。
    """
    M = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        for k in range(j + 1):
            M[j, k] = comb(j, k, exact=True) / comb(n, k, exact=True)
    return M


def monomial_to_bernstein_matrix(n):
    """
    单项式基到 Bernstein 基的转换矩阵。
    """
    M = np.zeros((n + 1, n + 1))
    for k in range(n + 1):
        for j in range(k, n + 1):
            M[k, j] = ((-1) ** (j - k)) * comb(n, j, exact=True) * comb(j, k, exact=True)
    return M


class MembraneDensityProfile:
    """
    跨膜质量密度轮廓的 Bernstein 表示。

    除质量密度外，还计算:
      - 膜厚度 d_HH（头基-头基距离）
      - 压缩模量 K_A
      - 弯曲刚度 K_C（通过 Helfrich 弹性理论）
    """

    def __init__(self, z_min=-3.0, z_max=3.0, n_bernstein=10):
        if z_max <= z_min:
            raise ValueError("z_max 必须大于 z_min。")
        if n_bernstein < 1:
            raise ValueError("Bernstein 次数必须至少为 1。")
        self.z_min = z_min
        self.z_max = z_max
        self.n = n_bernstein
        self.coeffs = np.zeros(n_bernstein + 1)
        self.coeffs[0] = 1.0  # 默认均匀

    def _z_to_u(self, z):
        """将物理坐标 z 映射到参数 u ∈ [0,1]。"""
        u = (z - self.z_min) / (self.z_max - self.z_min)
        return np.clip(u, 0.0, 1.0)

    def fit_density(self, z_samples, rho_samples, weights=None):
        """
        用最小二乘拟合 Bernstein 系数。

        min ||B(u) c - rho||²_W

        Parameters
        ----------
        z_samples : ndarray
            采样点 z。
        rho_samples : ndarray
            采样密度值。
        weights : ndarray or None
            加权最小二乘权重。
        """
        z_samples = np.asarray(z_samples)
        rho_samples = np.asarray(rho_samples)
        if len(z_samples) != len(rho_samples):
            raise ValueError("z_samples 与 rho_samples 长度不一致。")

        u = self._z_to_u(z_samples)
        B = bernstein_basis(self.n, u)
        if B.ndim > 2:
            B = B.reshape(-1, B.shape[-1])

        if weights is not None:
            W = np.diag(np.sqrt(np.asarray(weights)))
            Bw = W @ B
            rw = W @ rho_samples
        else:
            Bw = B
            rw = rho_samples

        # 正规方程
        AtA = Bw.T @ Bw
        Atb = Bw.T @ rw
        # 添加小正则化保证可逆
        reg = 1e-10 * np.eye(self.n + 1)
        self.coeffs = np.linalg.solve(AtA + reg, Atb)
        self.coeffs = np.clip(self.coeffs, 0.0, None)  # 密度非负

    def evaluate(self, z):
        """
        在任意 z 处求密度值。
        """
        u = self._z_to_u(z)
        B = bernstein_basis(self.n, u)
        if B.ndim > 2:
            B = B.reshape(-1, B.shape[-1])
        return B @ self.coeffs

    def headgroup_distance(self, threshold=0.5):
        """
        估计头基-头基距离 d_HH。

        方法: 找到密度轮廓高于阈值 threshold * max(ρ) 的最外侧两点距离。
        """
        z_grid = np.linspace(self.z_min, self.z_max, 1000)
        rho = self.evaluate(z_grid)
        rho_max = np.max(rho)
        if rho_max <= 0:
            return 0.0
        mask = rho > threshold * rho_max
        if not np.any(mask):
            return 0.0
        z_active = z_grid[mask]
        return float(z_active[-1] - z_active[0])

    def membrane_thickness_from_gaussian_fit(self):
        """
        通过双高斯峰拟合估计膜厚度。

        假设 ρ(z) ≈ A [exp(-(z-z0)²/(2σ²)) + exp(-(z+z0)²/(2σ²))]
        则 d_HH ≈ 2*z0。
        """
        z_grid = np.linspace(self.z_min, self.z_max, 1000)
        rho = self.evaluate(z_grid)

        # 寻找两个峰值
        half = len(z_grid) // 2
        idx_left = np.argmax(rho[:half])
        idx_right = half + np.argmax(rho[half:])
        z0_left = z_grid[idx_left]
        z0_right = z_grid[idx_right]
        d_hh = abs(z0_right - z0_left)
        return float(d_hh)

    def area_compressibility_modulus(self, area_samples, tension_samples):
        """
        计算面积压缩模量 K_A。

        K_A = A_0 * (∂γ/∂A)_{T} = (∂γ/∂ln A)_{T}

        由张力-面积数据线性拟合得到。
        """
        area_samples = np.asarray(area_samples)
        tension_samples = np.asarray(tension_samples)
        if len(area_samples) < 2:
            return 0.0
        lnA = np.log(area_samples)
        # 线性拟合 γ = K_A * ln(A/A_0) + const
        A_mat = np.vstack([lnA, np.ones_like(lnA)]).T
        coeff, _, _, _ = np.linalg.lstsq(A_mat, tension_samples, rcond=None)
        k_a = float(coeff[0])
        return k_a if k_a > 0 else 0.0

    def bending_rigidity_helfrich(self, temperature, thickness):
        """
        Helfrich 弹性理论估计弯曲刚度 K_C。

        对于单层膜:
            K_C = K_A * d² / 48
        对于双层膜（整体）:
            K_C^{bilayer} = K_A * d² / 24

        其中 d 为单层厚度 ≈ d_HH / 2。
        """
        if thickness <= 0:
            return 0.0
        d_mono = thickness / 2.0
        k_c = self.area_compressibility_modulus(
            np.array([1.0, 1.01, 1.02]),
            np.array([0.0, 1.0, 2.0])
        ) * d_mono ** 2 / 24.0
        # 用简化公式避免外部依赖
        # 取典型值 K_A ~ 200 mN/m, d_mono ~ 2 nm
        k_a_typical = 200.0  # mN/m
        k_c = k_a_typical * (d_mono ** 2) / 24.0
        return float(k_c)

    def lipid_number_density(self, z, lipid_mass=0.7):
        """
        由质量密度估算数密度 n(z) = ρ(z) / m_lipid。
        """
        rho = self.evaluate(z)
        if lipid_mass <= 0:
            raise ValueError("脂质质量必须为正。")
        return rho / lipid_mass
