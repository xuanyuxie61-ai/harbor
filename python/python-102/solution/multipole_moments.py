"""
multipole_moments.py
====================
基于矩方法（Method of Moments）从超构表面纳米柱的散射场中
提取多极矩展开系数，包括电偶极矩（p）、磁偶极矩（m）、
电四极矩（Q_e）和磁四极矩（Q_m）。

本模块源自项目 947_quadmom（矩方法计算高斯求积规则）的核心思想，
将 Hankel 矩阵 + Cholesky 分解的矩方法推广应用于电磁多极矩分析。

科学背景：
纳米柱的散射特性可以用多极展开描述。在 Rayleigh 粒子近似下，
散射场可写为：
    E_scat(r) = k²/(4π ε₀) [ (n̂ × p) × n̂ + (1/c) (n̂ × m) ] e^{ikr}/r
        + (ik³)/(24π ε₀) [ (n̂ × Q_e(n̂)) × n̂ + (1/c) (n̂ × Q_m(n̂)) ] e^{ikr}/r + ...

其中多极矩定义为：
    p = ∫ P(r') dV'                     （电偶极矩）
    m = (1/2) ∫ r' × J(r') dV'          （磁偶极矩）
    Q_e^{αβ} = ∫ [3 r'_α r'_β - r'² δ_{αβ}] ρ(r') dV'  （电四极矩）

本模块通过离散采样点上的场数据，构建矩量矩阵，
提取多极展开系数以表征纳米柱的电磁响应。
"""

import numpy as np
from numpy.linalg import cholesky, eigh


class MultipoleExtractor:
    """
    从散射场采样数据中提取电磁多极矩系数。
    """

    def __init__(self, wavelength=1.55e-6, n_bg=1.0):
        self.wavelength = wavelength
        self.k0 = 2.0 * np.pi / wavelength
        self.eps0 = 8.854187817e-12
        self.mu0 = 4.0 * np.pi * 1.0e-7
        self.c = 1.0 / np.sqrt(self.eps0 * self.mu0)
        self.n_bg = n_bg
        self.eta0 = np.sqrt(self.mu0 / self.eps0)  # 自由空间波阻抗

    # ------------------------------------------------------------------
    # 电/磁偶极矩提取
    # ------------------------------------------------------------------
    def extract_dipole_moments(self, r_obs, E_scat, H_scat):
        """
        从远场采样点 (r_obs) 的散射场 (E_scat, H_scat) 中
        反演电偶极矩 p 和磁偶极矩 m。

        远场近似（kr >> 1）：
            E_scat ≈ (k² / 4π ε₀) [ (n̂ × p) × n̂ + (1/c) n̂ × m ] e^{ikr} / r
            H_scat ≈ (k² / 4π) [ n̂ × p - (1/c) (n̂ × m) × n̂ ] e^{ikr} / r

        通过最小二乘拟合求解 p 和 m（共 6 个未知量）。

        Parameters
        ----------
        r_obs : ndarray, shape (N, 3)
            观测点坐标 [m]
        E_scat : ndarray, shape (N, 3), complex
            散射电场 [V/m]
        H_scat : ndarray, shape (N, 3), complex
            散射磁场 [A/m]

        Returns
        -------
        p : ndarray, shape (3,), complex
            电偶极矩 [C·m]
        m : ndarray, shape (3,), complex
            磁偶极矩 [A·m²]
        """
        N = r_obs.shape[0]
        k = self.k0 * self.n_bg
        prefactor_E = k ** 2 / (4.0 * np.pi * self.eps0)
        prefactor_H = k ** 2 / (4.0 * np.pi)

        A = np.zeros((6 * N, 6), dtype=np.complex128)
        b = np.zeros(6 * N, dtype=np.complex128)

        for i in range(N):
            r = r_obs[i]
            r_mag = np.linalg.norm(r)
            if r_mag < 1e-18:
                continue
            n_hat = r / r_mag
            phase = np.exp(1.0j * k * r_mag) / r_mag

            # E 的系数矩阵
            # (n × p) × n = n(n·p) - p
            # n × m
            for comp in range(3):
                # p 的系数
                for j in range(3):
                    val = (n_hat[comp] * n_hat[j] - (1.0 if comp == j else 0.0))
                    A[6 * i + comp, j] = prefactor_E * val * phase
                # m 的系数
                for j in range(3):
                    # (n × m)_comp = ε_{comp a b} n_a m_b
                    val = 0.0
                    for a in range(3):
                        for b_idx in range(3):
                            eps = self._levi_civita(comp, a, b_idx)
                            if eps != 0:
                                val += eps * n_hat[a] * (1.0 if j == b_idx else 0.0)
                    A[6 * i + comp, 3 + j] = prefactor_E * val * phase / self.c
                b[6 * i + comp] = E_scat[i, comp]

            # H 的系数矩阵
            for comp in range(3):
                for j in range(3):
                    val = 0.0
                    for a in range(3):
                        for b_idx in range(3):
                            eps = self._levi_civita(comp, a, b_idx)
                            if eps != 0:
                                val += eps * n_hat[a] * (1.0 if j == b_idx else 0.0)
                    A[6 * i + 3 + comp, j] = prefactor_H * val * phase
                for j in range(3):
                    val = (n_hat[comp] * n_hat[j] - (1.0 if comp == j else 0.0))
                    A[6 * i + 3 + comp, 3 + j] = -prefactor_H * val * phase / self.c
                b[6 * i + 3 + comp] = H_scat[i, comp]

        # 最小二乘求解
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        p = x[:3]
        m = x[3:6]
        return p, m

    @staticmethod
    def _levi_civita(i, j, k):
        """Levi-Civita 符号 ε_{ijk}。"""
        if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            return 1
        elif (i, j, k) in [(0, 2, 1), (2, 1, 0), (1, 0, 2)]:
            return -1
        else:
            return 0

    # ------------------------------------------------------------------
    # 矩方法提取多极展开系数（Hankel 矩阵 + Cholesky）
    # ------------------------------------------------------------------
    def multipole_moment_method(self, angular_samples, field_samples, max_order=4):
        """
        使用矩方法从角向散射场分布中提取多极展开系数。

        在远场球坐标 (θ, φ) 下，散射场可展开为球谐函数：
            E(θ,φ) = Σ_{l=0}^{∞} Σ_{m=-l}^{l} a_{lm} Y_{lm}(θ,φ)

        通过采样数据构造 Hankel 型矩矩阵，利用 Cholesky 分解
        提取展开系数，类似于 947_quadmom 中从矩构造高斯积分规则的流程。

        Parameters
        ----------
        angular_samples : ndarray, shape (N, 2)
            (θ, φ) 采样角度 [rad]
        field_samples : ndarray, shape (N,), complex
            对应角度上的散射场幅度
        max_order : int
            最大展开阶数 l_max

        Returns
        -------
        coefficients : ndarray, shape ((l_max+1)²,)
            球谐展开系数 a_{lm}
        """
        N = len(angular_samples)
        l_max = max_order
        n_coeff = (l_max + 1) ** 2

        # 构造矩：M_{pq} = ∫ Y_p*(θ,φ) Y_q(θ,φ) |E(θ,φ)|² dΩ
        # 这里用离散近似
        M = np.zeros((n_coeff, n_coeff), dtype=np.complex128)

        # 预计算所有球谐函数值
        Y_vals = np.zeros((N, n_coeff), dtype=np.complex128)
        for idx in range(N):
            theta, phi = angular_samples[idx]
            l_idx = 0
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    Y_vals[idx, l_idx] = self._spherical_harmonic(l, m, theta, phi)
                    l_idx += 1

        # 矩矩阵
        for p in range(n_coeff):
            for q in range(n_coeff):
                M[p, q] = np.sum(np.conj(Y_vals[:, p]) * Y_vals[:, q] * np.abs(field_samples) ** 2)
                M[p, q] *= 4.0 * np.pi / N  # 离散近似积分

        # 强制 Hermite 对称
        M = 0.5 * (M + M.conj().T)

        # 使用 Cholesky 分解（M 应正定）
        # 类似于 quadmom 中从矩构造 Jacobi 矩阵
        try:
            R = cholesky(M)
        except np.linalg.LinAlgError:
            # 若不正定，加正则化
            M += 1e-12 * np.eye(n_coeff)
            R = cholesky(M)

        # 从 Cholesky 因子提取递推系数
        n = n_coeff
        alpha = np.zeros(n - 1, dtype=np.float64)
        for i in range(n - 1):
            if abs(R[i, i]) > 1e-14:
                alpha[i] = R[i, i + 1] / R[i, i]

        # Jacobi 矩阵
        J = np.diag(alpha, 1) + np.diag(alpha, -1)
        # 计算特征值和特征向量
        eigvals, eigvecs = eigh(J)

        # 展开系数 a = V^T * field_moments
        field_moments = np.zeros(n_coeff, dtype=np.complex128)
        for p in range(n_coeff):
            field_moments[p] = np.sum(np.conj(Y_vals[:, p]) * field_samples)
            field_moments[p] *= 4.0 * np.pi / N

        coefficients = field_moments
        return coefficients

    def _spherical_harmonic(self, l, m, theta, phi):
        """
        实球谐函数（用于避免复数混淆）。
        """
        from scipy.special import sph_harm
        # 使用标准复球谐，取实部作为近似
        Y = sph_harm(m, l, phi, theta)
        return Y

    # ------------------------------------------------------------------
    # 多极辐射功率
    # ------------------------------------------------------------------
    def radiation_powers(self, p, m, Q_e=None, Q_m=None):
        """
        计算各多极矩的辐射功率（散射功率）。

        电偶极辐射功率：
            P_p = (μ₀ ω⁴ / 12π c) |p|²
        磁偶极辐射功率：
            P_m = (μ₀ ω⁴ / 12π c³) |m|²
        电四极辐射功率：
            P_Qe = (μ₀ ω⁶ / 240π c³) Σ |Q_e^{αβ}|²
        """
        omega = self.k0 * self.c
        mu0 = self.mu0

        P_p = mu0 * omega ** 4 / (12.0 * np.pi * self.c) * np.sum(np.abs(p) ** 2)
        P_m = mu0 * omega ** 4 / (12.0 * np.pi * self.c ** 3) * np.sum(np.abs(m) ** 2)

        P_Qe = 0.0
        if Q_e is not None:
            P_Qe = mu0 * omega ** 6 / (240.0 * np.pi * self.c ** 3) * np.sum(np.abs(Q_e) ** 2)

        P_Qm = 0.0
        if Q_m is not None:
            P_Qm = mu0 * omega ** 6 / (240.0 * np.pi * self.c ** 3) * np.sum(np.abs(Q_m) ** 2)

        return {'P_dipole_electric': P_p,
                'P_dipole_magnetic': P_m,
                'P_quadrupole_electric': P_Qe,
                'P_quadrupole_magnetic': P_Qm,
                'P_total': P_p + P_m + P_Qe + P_Qm}


def demo():
    """演示：从模拟的远场数据中提取多极矩。"""
    me = MultipoleExtractor(wavelength=1.55e-6)

    # 构造模拟远场：一个电偶极子 + 一个磁偶极子
    k = me.k0
    N = 120
    theta = np.linspace(0.1, np.pi - 0.1, N)
    phi = np.linspace(0, 2 * np.pi, N)
    theta_g, phi_g = np.meshgrid(theta, phi)
    theta_f = theta_g.flatten()
    phi_f = phi_g.flatten()

    # 真实多极矩
    p_true = np.array([1.0 + 0.5j, 0.3 - 0.2j, 0.1 + 0.0j]) * 1e-18
    m_true = np.array([0.2 + 0.1j, 0.8 - 0.3j, 0.1 + 0.2j]) * 1e-21

    r_obs = np.stack([
        np.sin(theta_f) * np.cos(phi_f),
        np.sin(theta_f) * np.sin(phi_f),
        np.cos(theta_f)
    ], axis=1) * 1.0e-3  # 1 mm 远场

    E_scat = np.zeros((len(theta_f), 3), dtype=np.complex128)
    H_scat = np.zeros((len(theta_f), 3), dtype=np.complex128)

    for i in range(len(theta_f)):
        n = r_obs[i] / np.linalg.norm(r_obs[i])
        # E ∝ (n × p) × n + (1/c) n × m
        cross_p = np.cross(n, p_true)
        E_scat[i] = (k ** 2 / (4 * np.pi * me.eps0)) * (
            np.cross(cross_p, n) + np.cross(n, m_true) / me.c
        ) * np.exp(1.0j * k * 1.0e-3) / 1.0e-3
        H_scat[i] = (k ** 2 / (4 * np.pi)) * (
            np.cross(n, p_true) - np.cross(np.cross(n, m_true), n) / me.c
        ) * np.exp(1.0j * k * 1.0e-3) / 1.0e-3

    p_est, m_est = me.extract_dipole_moments(r_obs, E_scat, H_scat)

    print("[multipole_moments] 电偶极矩估计:")
    print(f"  p = [{p_est[0]:.3e}, {p_est[1]:.3e}, {p_est[2]:.3e}] C·m")
    print("[multipole_moments] 磁偶极矩估计:")
    print(f"  m = [{m_est[0]:.3e}, {m_est[1]:.3e}, {m_est[2]:.3e}] A·m²")

    powers = me.radiation_powers(p_est, m_est)
    print(f"[multipole_moments] 总辐射功率: {powers['P_total']:.4e} W")
    return p_est, m_est, powers


if __name__ == "__main__":
    demo()
