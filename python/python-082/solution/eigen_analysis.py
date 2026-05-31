# -*- coding: utf-8 -*-
"""
eigen_analysis.py
=================
特征值分析与时间积分稳定性评估模块。

源自种子项目：
  - 1206_test_eigen（具有预设特征结构的测试矩阵生成）
  - 104_boundary_locus（ODE 方法稳定性区域边界分析）

科学背景：
---------
在复合材料损伤演化分析中，特征值问题出现在多个环节：

1. 结构模态分析（自由振动）：
   (K - ω² M) φ = 0
   其中 K 为刚度矩阵，M 为质量矩阵，ω 为圆频率，φ 为模态形状。
   固有频率决定了结构的动态响应特性，损伤导致刚度退化，
   从而固有频率降低（频率漂移是结构健康监测的重要指标）。

2. 损伤局部化稳定性：
   当损伤带形成时，切线刚度矩阵 K_tan 失去正定性，
   最小特征值变为负值，对应结构失稳（snap-back / snap-through）。
   失稳判据：det(K_tan) = 0，即最小特征值 λ_min = 0。

3. 时间积分稳定性：
   显式时间步进受 CFL 条件限制：
     Δt ≤ CFL / ρ(A)
   其中 ρ(A) 为空间离散算子的谱半径（最大特征值模）。
   对于 DG 方法，ρ(A) ∝ N² * c / h_min。

4. 稳定性区域分析（Boundary Locus 方法）：
   对测试方程 y' = λy，数值方法的放大因子 R(z) 满足 |R(z)| ≤ 1
   的区域为绝对稳定区域。Boundary Locus 通过求解 |R(z)| = 1
   的等值线得到稳定性边界。

核心公式：
  瑞利商：
    R(v) = (v^T K v) / (v^T M v)
    λ_min ≤ R(v) ≤ λ_max

  结构阻尼比：
    ζ_i = C_i / (2 * ω_i)
    其中 C_i = φ_i^T C φ_i / (φ_i^T M φ_i)

  LSERK(5,4) 放大因子：
    R(z) = 1 + z + z²/2 + z³/6 + z⁴/24 + z⁵/120 + z⁶/600
    （注意：经典 5 级 4 阶 LSERK 的 R(z) 实际上是 5 阶多项式）
"""

import numpy as np
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import csr_matrix
from typing import Optional, Tuple


class StructuralModalAnalysis:
    """
    复合材料结构的模态分析与频率漂移计算。
    """

    def __init__(self, K: csr_matrix, M: Optional[csr_matrix] = None):
        """
        Parameters
        ----------
        K : csr_matrix
            刚度矩阵 (n, n)。
        M : csr_matrix or None
            质量矩阵；None 时假设单位质量矩阵。
        """
        self.K = K
        self.n = K.shape[0]
        if M is None:
            from scipy.sparse import eye
            self.M = eye(self.n, format='csr')
        else:
            self.M = M

    def compute_modes(self, num_modes: int = 10,
                      sigma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算前 num_modes 个特征对 (λ, φ)。
        对广义特征值问题 K φ = λ M φ，λ = ω²。

        Returns
        -------
        eigenvalues : np.ndarray
            特征值 λ = ω² [rad²/s²]。
        eigenvectors : np.ndarray
            特征向量 φ，每列一个模态，已 M-归一化。
        """
        if num_modes >= self.n - 1:
            # 对小型问题退化为稠密求解
            K_dense = self.K.toarray()
            M_dense = self.M.toarray()
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M_dense, K_dense))
            # 取实部并排序
            eigenvalues = np.real(eigenvalues)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx[:num_modes]]
            eigenvectors = eigenvectors[:, idx[:num_modes]]
        else:
            # 对大型问题使用 Lanczos 迭代（ARPACK）
            # 由于 K 在损伤后可能不定，使用 shift-invert 模式
            if sigma is None:
                sigma = 0.0
            try:
                eigenvalues, eigenvectors = eigsh(self.K, k=num_modes,
                                                   M=self.M, sigma=sigma,
                                                   which='LM', mode='normal')
            except Exception:
                # 若失败，尝试不带 M 直接求解标准特征值
                eigenvalues, eigenvectors = eigsh(self.K, k=min(num_modes, self.n - 2),
                                                   which='SM')
        # 数值鲁棒性：去除负的小特征值（数值噪声）
        eigenvalues = np.where(eigenvalues < 1e-12, 0.0, eigenvalues)
        # M-归一化
        for i in range(eigenvectors.shape[1]):
            norm = np.sqrt(eigenvectors[:, i].T @ self.M @ eigenvectors[:, i])
            if norm > 1e-30:
                eigenvectors[:, i] /= norm
        return eigenvalues, eigenvectors

    def natural_frequencies(self, num_modes: int = 10) -> np.ndarray:
        """返回前 num_modes 个固有频率 [Hz]。"""
        eigenvalues, _ = self.compute_modes(num_modes)
        return np.sqrt(eigenvalues) / (2.0 * np.pi)

    def frequency_shift_due_to_damage(self, K_damaged: csr_matrix,
                                       num_modes: int = 5) -> np.ndarray:
        """
        计算损伤导致的频率漂移百分比。

        公式：
          Δf_i / f_i = (f_damaged,i - f_undamaged,i) / f_undamaged,i * 100%
        """
        f_0 = self.natural_frequencies(num_modes)
        damaged_analyzer = StructuralModalAnalysis(K_damaged, self.M)
        f_d = damaged_analyzer.natural_frequencies(num_modes)
        shift = (f_d - f_0) / (f_0 + 1e-30) * 100.0
        return shift

    def instability_index(self) -> float:
        """
        结构失稳指标：最小特征值。
        λ_min < 0 表示结构已失稳（刚度矩阵非正定）。
        """
        try:
            lam_min, _ = eigsh(self.K, k=1, which='SM')
            return float(lam_min[0])
        except Exception:
            # 退化为幂法估计最小特征值
            vals = np.linalg.eigvalsh(self.K.toarray())
            return float(np.min(vals))

    def modal_participation_factor(self, force_pattern: np.ndarray,
                                    num_modes: int = 10) -> np.ndarray:
        """
        模态参与因子：衡量各模态对外力激励的贡献。

        公式：
          P_i = (φ_i^T F) / (φ_i^T M φ_i) = φ_i^T F   （因已 M-归一化）
        """
        _, phi = self.compute_modes(num_modes)
        return phi.T @ force_pattern


class TestMatrixGenerator:
    """
    具有预设特征结构的测试矩阵生成器（源自 test_eigen）。
    用于验证特征值求解器的正确性。
    """

    @staticmethod
    def symmetric_with_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
        """
        生成对称矩阵 A = Q Λ Q^T，其中 Q 为随机正交矩阵，Λ = diag(eigenvalues)。
        """
        n = len(eigenvalues)
        # 随机正交矩阵：QR 分解
        X = np.random.randn(n, n)
        Q, _ = np.linalg.qr(X)
        Lambda = np.diag(eigenvalues)
        A = Q @ Lambda @ Q.T
        return A

    @staticmethod
    def nonsymmetric_with_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
        """
        生成非对称矩阵 A = Q^T T Q，其中 T 为具有指定特征值的上三角 Schur 型。
        """
        n = len(eigenvalues)
        T = np.triu(np.random.randn(n, n), k=1)
        np.fill_diagonal(T, eigenvalues)
        X = np.random.randn(n, n)
        Q, _ = np.linalg.qr(X)
        A = Q.T @ T @ Q
        return A

    @staticmethod
    def damaged_stiffness_spectrum(n: int, damage_level: float) -> np.ndarray:
        """
        模拟损伤刚度矩阵的特征值谱：
        无损时特征值均匀分布；损伤时低阶模态（长波长）受更大影响。
        """
        lam_base = np.linspace(1.0, n ** 2, n)
        # 损伤对低阶模态影响更大（整体弯曲刚度降低）
        damage_factor = 1.0 - damage_level * np.exp(-np.arange(n) / (n / 5.0))
        damage_factor = np.clip(damage_factor, 0.1, 1.0)
        return lam_base * damage_factor


class StabilityRegionAnalysis:
    """
    时间积分方法的稳定性区域分析（源自 boundary_locus）。
    """

    @staticmethod
    def amplification_factor_lserk45(z: complex) -> complex:
        """
        5 级 4 阶低存储 RK（LSERK45）的放大因子 R(z)。
        对于 RK 方法，R(z) = 1 + z b^T (I - z A)^{-1} 1。
        LSERK45 的 Butcher 表对应的 R(z) 为 5 阶多项式近似。
        这里使用其经典形式：
          R(z) ≈ 1 + z + z²/2 + z³/6 + z⁴/24 + z⁵/120
        """
        return (1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0
                + z ** 4 / 24.0 + z ** 5 / 120.0)

    @staticmethod
    def amplification_factor_explicit_euler(z: complex) -> complex:
        """显式 Euler：R(z) = 1 + z。"""
        return 1.0 + z

    @staticmethod
    def amplification_factor_implicit_euler(z: complex) -> complex:
        """隐式 Euler：R(z) = 1 / (1 - z)。"""
        return 1.0 / (1.0 - z + 1e-30)

    @staticmethod
    def compute_boundary_locus(method: str = 'lserk45',
                                num_points: int = 400,
                                z_max: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算稳定性边界 |R(z)| = 1 的曲线参数。
        采用极坐标扫描：对 θ∈[0, 2π)，通过二分法找 r 使得 |R(r*exp(iθ))| = 1。

        Returns
        -------
        x, y : 边界点坐标（复平面上 z = x + iy）。
        """
        if method == 'lserk45':
            R = StabilityRegionAnalysis.amplification_factor_lserk45
        elif method == 'explicit_euler':
            R = StabilityRegionAnalysis.amplification_factor_explicit_euler
        elif method == 'implicit_euler':
            R = StabilityRegionAnalysis.amplification_factor_implicit_euler
        else:
            raise ValueError(f"Unknown method: {method}")

        thetas = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        x_boundary = np.zeros(num_points)
        y_boundary = np.zeros(num_points)

        for i, theta in enumerate(thetas):
            # 二分法找 r
            r_low, r_high = 0.0, z_max * 2.0
            for _ in range(50):
                r_mid = (r_low + r_high) / 2.0
                z = r_mid * np.exp(1j * theta)
                mag = abs(R(z))
                if mag > 1.0:
                    r_high = r_mid
                else:
                    r_low = r_mid
            r_opt = (r_low + r_high) / 2.0
            z_bound = r_opt * np.exp(1j * theta)
            x_boundary[i] = z_bound.real
            y_boundary[i] = z_bound.imag

        return x_boundary, y_boundary

    @staticmethod
    def cfl_limit_estimate(wave_speed: float, dx_min: float,
                           poly_order: int, method: str = 'lserk45') -> float:
        """
        基于稳定性区域估计 DG 谱元的最大允许时间步。

        公式：
          CFL = min_{z on boundary} |z| / (N² * c / dx_min)
        对 LSERK45，经验 CFL ≈ 0.5 / (N^1.5)。
        """
        if method == 'lserk45':
            cfl_coeff = 0.5 / (poly_order ** 1.5 + 1e-30)
        elif method == 'explicit_euler':
            cfl_coeff = 0.1 / (poly_order ** 2 + 1e-30)
        else:
            cfl_coeff = 1e6  # 隐式方法无 CFL 限制
        return cfl_coeff * dx_min / (wave_speed + 1e-30)


if __name__ == "__main__":
    # 自测试 1：测试矩阵生成与验证
    eigenvals = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    A_sym = TestMatrixGenerator.symmetric_with_eigenvalues(eigenvals)
    computed_eigs = np.sort(np.linalg.eigvalsh(A_sym))
    print("Symmetric test matrix eigenvalues:", computed_eigs)
    assert np.allclose(computed_eigs, np.sort(eigenvals), atol=1e-10)

    # 自测试 2：稳定性边界
    x_b, y_b = StabilityRegionAnalysis.compute_boundary_locus('lserk45', num_points=100)
    print("LSERK45 stability boundary sample:", x_b[0], y_b[0])

    # 自测试 3：模态分析
    from stiffness_matrix import StiffnessMatrixAssembler1D
    nodes = np.linspace(0.0, 1.0, 51)
    assembler = StiffnessMatrixAssembler1D(nodes, A=1e-4, E0=100e9)
    K = assembler.assemble_global_stiffness()
    modal = StructuralModalAnalysis(K)
    freqs = modal.natural_frequencies(num_modes=5)
    print("Natural frequencies (Hz):", freqs)
    print("Instability index:", modal.instability_index())
