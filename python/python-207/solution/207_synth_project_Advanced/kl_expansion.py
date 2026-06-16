"""
kl_expansion.py — Karhunen-Loève 展开与随机场生成

科学背景
========
Karhunen-Loève 展开是随机过程的最优正交分解.
对于均值为零的二阶随机过程 Z(x,ω), 其 KL 展开为:

    Z(x, ω) = Σ_{k=1}^{∞} √λ_k · φ_k(x) · ξ_k(ω)

其中:
- λ_k, φ_k 为协方差算子的特征值和特征函数:
    ∫ C(x,x') φ_k(x') dx' = λ_k · φ_k(x)
- ξ_k(ω) 为互不相关随机变量, E[ξ_k]=0, Var[ξ_k]=1

截断到 K 项:
    Z_K(x, ω) = Σ_{k=1}^{K} √λ_k · φ_k(x) · ξ_k(ω)

对数正态随机场 (保证正扩散系数):
    κ(x, ω) = κ₀ · exp( σ_ln · Z_K(x,ω) / σ_Z )

算法来源 (种子项目 164_chebyshev1_exactness)
==============================================
利用 Gauss-Chebyshev 求积计算特征值问题的内积:
    λ_k = ∫∫ C(x,x') φ_k(x) φ_k(x') dx dx'
近似为:
    λ_k ≈ Σ_{j=1}^{n_q} w_j · C(x_j, ·) · φ_k(x_j)

其中 (x_j, w_j) 为 Gauss-Chebyshev 节点和权重.

核心公式
========
1. 离散特征值分解:  Σ · v_k = λ_k · v_k
2. 归一化特征向量:  φ_k(x_i) = v_k[i] / √(Σ_j v_k[j]² · Δx)
3. 截断误差:  ε_K = 1 - Σ_{k=1}^{K} λ_k / Σ_{all} λ_k
4. 随机场实现:  Z(x_i, ω) = Σ_{k=1}^{K} √λ_k · φ_k(x_i) · ξ_k(ω)
"""

import numpy as np
from covariance_cholesky import CovarianceKernel, cholesky_as006


class KLExpansion:
    """Karhunen-Loève 展开引擎.

    对给定协方差核和离散网格, 计算截断 KL 展开并生成随机场实现.
    """

    def __init__(self, kernel, x_grid, n_modes, gc_nodes=None, gc_weights=None):
        """
        参数
        ----
        kernel : CovarianceKernel
            协方差核对象
        x_grid : ndarray, shape (N,)
            空间网格节点
        n_modes : int
            KL 截断阶数 K
        gc_nodes, gc_weights : ndarray or None
            Gauss-Chebyshev 求积节点和权重 (用于精确特征值计算)
        """
        self.kernel = kernel
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.n_grid = len(self.x_grid)
        self.n_modes = n_modes
        self.dx = self.x_grid[1] - self.x_grid[0] if self.n_grid > 1 else 1.0

        # Gauss-Chebyshev 求积 (种子 164)
        if gc_nodes is not None and gc_weights is not None:
            self.gc_nodes = np.asarray(gc_nodes)
            self.gc_weights = np.asarray(gc_weights)
        else:
            n_q = min(30, self.n_grid)
            k = np.arange(1, n_q + 1)
            self.gc_nodes = np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n_q))
            self.gc_weights = np.full(n_q, np.pi / n_q)

        # 计算特征分解
        self._compute_eigendecomposition()

    def _compute_eigendecomposition(self):
        """计算离散协方差矩阵的特征分解.

        1. 构造 Σ[i,j] = C(x_i, x_j)
        2. 特征值分解: Σ = V · diag(λ) · V^T
        3. 截取前 K 个最大特征值及对应特征向量
        4. 用 Gauss-Chebyshev 求积校正特征值
        """
        # 构造协方差矩阵
        self.cov_matrix = self.kernel.evaluate(self.x_grid)

        # 对称化 (消除浮点不对称)
        self.cov_matrix = 0.5 * (self.cov_matrix + self.cov_matrix.T)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)

        # eigh 返回升序, 转为降序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 截取前 K 个模式
        K = min(self.n_modes, self.n_grid)
        self.eigenvalues = np.maximum(eigenvalues[:K], 0.0)  # 非负约束
        self.eigenvectors = eigenvectors[:, :K]

        # 归一化特征函数 (L² 归一化):
        #   ∫ φ_k(x)² dx ≈ Σ_i φ_k(x_i)² · Δx = 1
        norms = np.sqrt(np.sum(self.eigenvectors ** 2, axis=0) * self.dx)
        norms = np.maximum(norms, 1.0e-30)
        self.eigenfunctions = self.eigenvectors / norms[None, :]

        # 用 Gauss-Chebyshev 求积校正特征值 (种子 164):
        #   λ_k = ∫∫ C(x,x') φ_k(x) φ_k(x') dx dx'
        #     ≈ Σ_j w_j · [Σ_i w_i · C(x_j, x_i) · φ_k(x_i)] · φ_k(x_j)
        self._correct_eigenvalues_with_quadrature()

        # 累计能量
        total_energy = np.sum(np.maximum(eigenvalues, 0.0))
        kept_energy = np.sum(self.eigenvalues)
        self.energy_retained = kept_energy / total_energy if total_energy > 0 else 0.0

    def _correct_eigenvalues_with_quadrature(self):
        """利用 Gauss-Chebyshev 求积精确计算特征值.

        对于平方指数核, 特征值的解析表达为:
            λ_k = (2πℓ²)^{1/2} · exp(-k²π²ℓ²/(2L²))  (近似)

        这里用数值求积验证:
            λ_k ≈ (L/2) · Σ_j w_j · [C · φ_k](x_j) · φ_k(x_j)
        """
        # 将 GC 节点映射到 [0, L]
        L = self.x_grid[-1] - self.x_grid[0]
        x_mid = 0.5 * (self.x_grid[0] + self.x_grid[-1])
        gc_mapped = x_mid + 0.5 * L * self.gc_nodes
        w_mapped = 0.5 * L * self.gc_weights

        # 在 GC 节点上计算协方差
        C_gc = self.kernel.evaluate(gc_mapped)

        # 插值特征函数到 GC 节点
        from numpy.polynomial.chebyshev import chebval
        phi_at_gc = np.zeros((len(gc_mapped), self.n_modes))
        for k in range(self.n_modes):
            # 用最近邻插值 (简单方案)
            for j, xg in enumerate(gc_mapped):
                idx_nearest = np.argmin(np.abs(self.x_grid - xg))
                phi_at_gc[j, k] = self.eigenfunctions[idx_nearest, k]

        # 数值积分: λ_k ≈ Σ_j w_j · [Σ_i w_i · C(x_j,x_i) · φ_k(x_i)] · φ_k(x_j)
        for k in range(self.n_modes):
            inner = C_gc @ (w_mapped * phi_at_gc[:, k])
            lambda_k_quad = np.sum(w_mapped * inner * phi_at_gc[:, k])
            # 混合: 取离散特征值和求积结果的加权平均
            lambda_k_discrete = self.eigenvalues[k]
            if lambda_k_quad > 0:
                self.eigenvalues[k] = 0.7 * lambda_k_discrete + 0.3 * lambda_k_quad

    def generate_realization(self, xi_coeffs):
        """生成随机场的一个实现.

        Z_K(x_i, ω) = Σ_{k=1}^{K} √λ_k · φ_k(x_i) · ξ_k

        参数
        ----
        xi_coeffs : ndarray, shape (K,)
            KL 系数 (标准正态随机变量)

        返回
        ----
        z_field : ndarray, shape (N,)
            随机场在网格上的值
        """
        xi_coeffs = np.asarray(xi_coeffs, dtype=float)
        if len(xi_coeffs) != self.n_modes:
            raise ValueError(f"需要 {self.n_modes} 个 KL 系数, 得到 {len(xi_coeffs)}")

        # Z(x_i) = Σ_k sqrt(λ_k) · φ_k(x_i) · ξ_k
        sqrt_lambda = np.sqrt(self.eigenvalues)
        z_field = self.eigenfunctions @ (sqrt_lambda * xi_coeffs)
        return z_field

    def generate_lognormal_field(self, xi_coeffs, kappa_0, sigma_kappa):
        """生成对数正态随机扩散系数场.

        保证 κ > 0:
            κ(x,ω) = κ₀ · exp( σ_ln/σ_Z · Z_K(x,ω) )

        其中 σ_ln 通过匹配方差确定:
            σ_ln² = ln(1 + (σ_κ/κ₀)²)

        参数
        ----
        xi_coeffs : ndarray, shape (K,)
        kappa_0 : float  基准扩散系数
        sigma_kappa : float  κ 的标准差

        返回
        ----
        kappa_field : ndarray, shape (N,)
        """
        z_field = self.generate_realization(xi_coeffs)
        z_std = np.sqrt(np.sum(self.eigenvalues))
        z_std = max(z_std, 1.0e-30)

        # 对数正态参数
        cv = sigma_kappa / kappa_0  # 变异系数
        sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))

        kappa_field = kappa_0 * np.exp(sigma_ln * z_field / z_std - 0.5 * sigma_ln ** 2)
        return kappa_field

    def generate_batch(self, xi_matrix, kappa_0=None, sigma_kappa=None):
        """批量生成随机场实现.

        参数
        ----
        xi_matrix : ndarray, shape (n_samples, K)
        kappa_0, sigma_kappa : float or None
            若提供, 则生成对数正态场

        返回
        ----
        fields : ndarray, shape (n_samples, N)
        """
        n_samples = xi_matrix.shape[0]
        fields = np.zeros((n_samples, self.n_grid))
        for m in range(n_samples):
            if kappa_0 is not None and sigma_kappa is not None:
                fields[m] = self.generate_lognormal_field(
                    xi_matrix[m], kappa_0, sigma_kappa
                )
            else:
                fields[m] = self.generate_realization(xi_matrix[m])
        return fields
