"""
cdf_discrete_sampler.py — 基于离散 CDF 的分层随机场采样

科学背景
========
从连续随机场的离散近似中采样, 等价于从离散联合 PDF 中生成样本.
设 {ξ_i} 为标准独立随机变量, 则随机场的一个实现为:

    κ(x, ω) = κ₀ + σ_κ · L_c · F^{-1}(u)

其中 u = (u_1,...,u_N) 为从离散 CDF 中抽取的均匀随机向量,
F^{-1} 为逆 CDF 变换 (概率积分变换).

算法来源 (种子项目 291_discrete_pdf_sample_2d)
==============================================
核心思想:
1. 从 2D 离散 PDF 构造累积分布函数 CDF
2. 给定均匀随机数 U ~ Uniform(0,1), 通过逆 CDF 查找
   对应的格子单元 (i, j)
3. 在选定单元内进行二次随机采样 (dithering)

本项目中的角色
==============
将 2D PDF 采样推广到随机场的分层采样:
- 第一层: 沿 KL 模式方向进行 CDF 分层
- 第二层: 在每层内进行二次均匀采样
- 这比简单 MC 采样具有更好的覆盖率 (stratification)

核心公式
========
1. CDF 累积:  F[k] = Σ_{j≤k} p[j],  F[N]=1
2. 逆采样:  给定 U ~ U(0,1),  找 k 使 F[k-1] ≤ U < F[k]
3. 层内插值:  ξ = F_inv(U) + (r - 0.5)·Δ,  r ~ U(0,1)
4. 方差缩减效率:  Var_strat ≤ Var_mc / K  (K = 层数)
"""

import numpy as np


class DiscreteCDFSampler1D:
    """一维离散 CDF 采样器.

    给定概率向量 p = (p_1,...,p_K) (Σp_i = 1),
    构造 CDF 并实现逆 CDF 采样.
    """

    def __init__(self, pdf):
        """构造 CDF.

        参数
        ----
        pdf : ndarray, shape (K,)
            离散概率密度, 须非负且和为 1
        """
        pdf = np.asarray(pdf, dtype=float)
        if np.any(pdf < 0):
            raise ValueError("PDF 含有负值")
        total = np.sum(pdf)
        if total <= 0:
            raise ValueError("PDF 总和为零")
        self.pdf = pdf / total  # 归一化
        self.cdf = np.cumsum(self.pdf)
        self.cdf[-1] = 1.0  # 确保精确为 1
        self.n_bins = len(self.pdf)

    def inverse_cdf_lookup(self, u_values):
        """逆 CDF 查找: 对每个 u ∈ [0,1), 返回对应区间索引.

        算法 (种子 291):
            对每个 u, 找最小 k 使得 CDF[k] ≥ u

        参数
        ----
        u_values : ndarray
            均匀随机数 ∈ [0, 1)

        返回
        ----
        indices : ndarray of int
            区间索引 (0-based)
        """
        u_values = np.asarray(u_values, dtype=float)
        u_clipped = np.clip(u_values, 0.0, 1.0 - 1.0e-15)
        # searchsorted: 找第一个 ≥ u 的位置
        indices = np.searchsorted(self.cdf, u_clipped, side='left')
        indices = np.clip(indices, 0, self.n_bins - 1)
        return indices

    def sample_with_dithering(self, n_samples, rng=None):
        """带层内二次采样 (dithering) 的分层采样.

        将 [0,1] 分为 n_samples 个等宽层, 每层内均匀随机采样,
        然后通过逆 CDF 转换为离散索引.

        返回
        ----
        indices : ndarray, shape (n_samples,)
        u_values : ndarray, shape (n_samples,)
            实际使用的均匀随机数 (可用于重现)
        """
        if rng is None:
            rng = np.random.default_rng(42)
        # 分层: u_k = (k + r_k) / n_samples,  r_k ~ U(0,1)
        k = np.arange(n_samples)
        r = rng.uniform(0.0, 1.0, size=n_samples)
        u_values = (k + r) / n_samples
        indices = self.inverse_cdf_lookup(u_values)
        return indices, u_values

    def quantile_function(self, p):
        """分位数函数: Q(p) = min{k : CDF[k] ≥ p}.

        用于将均匀分布转换为离散分布.
        """
        return self.inverse_cdf_lookup(np.atleast_1d(p))


class DiscreteCDFSampler2D:
    """二维离散 CDF 采样器 (直接移植种子 291).

    将 2D PDF p[i,j] 展平为 1D 并按行优先序累积.
    采样时在选定单元 (i,j) 内进行二次均匀采样.
    """

    def __init__(self, pdf_2d):
        """
        参数
        ----
        pdf_2d : ndarray, shape (n1, n2)
        """
        pdf_2d = np.asarray(pdf_2d, dtype=float)
        if np.any(pdf_2d < 0):
            raise ValueError("2D PDF 含有负值")
        total = np.sum(pdf_2d)
        if total <= 0:
            raise ValueError("2D PDF 总和为零")
        self.pdf_2d = pdf_2d / total
        self.n1, self.n2 = pdf_2d.shape

        # 按行优先序展平并构造 CDF
        self.pdf_flat = self.pdf_2d.ravel()
        self.cdf_flat = np.cumsum(self.pdf_flat)
        self.cdf_flat[-1] = 1.0

    def sample(self, n_samples, rng=None):
        """从 2D 离散分布中采样 n_samples 个点.

        算法:
        1. 从 CDF_flat 中逆采样得到展平索引 k
        2. 转换为 (i, j) = (k // n2, k % n2)
        3. 在单元内加二次随机偏移

        返回
        ----
        ij_samples : ndarray, shape (n_samples, 2)
            连续坐标 (i + r1, j + r2) / (n1, n2)
        """
        if rng is None:
            rng = np.random.default_rng(42)

        u = rng.uniform(0.0, 1.0, size=n_samples)
        flat_idx = np.searchsorted(self.cdf_flat, u, side='left')
        flat_idx = np.clip(flat_idx, 0, len(self.pdf_flat) - 1)

        i_idx = flat_idx // self.n2
        j_idx = flat_idx % self.n2

        # 单元内二次采样 (dithering)
        r1 = rng.uniform(0.0, 1.0, size=n_samples)
        r2 = rng.uniform(0.0, 1.0, size=n_samples)

        xy = np.zeros((n_samples, 2))
        xy[:, 0] = (i_idx + r1) / self.n1
        xy[:, 1] = (j_idx + r2) / self.n2
        return xy


def stratified_kl_sampling(n_modes, n_samples, rng=None):
    """KL 模式系数的分层采样.

    对于 K 个独立标准正态 KL 系数 ξ_1,...,ξ_K,
    使用分层采样减少方差:
    - 将每个 ξ_k 的 CDF Φ 分为 n_strata 层
    - 拉丁超立方采样 (LHS) 确保均匀覆盖

    公式
    ----
    ξ_k = Φ^{-1}( (rank_k + U_k) / n_samples )
    其中 U_k ~ Uniform(0,1),  rank_k 为随机排列

    参数
    ----
    n_modes : int  (KL 截断阶数 K)
    n_samples : int  (MC 样本量)

    返回
    ----
    xi : ndarray, shape (n_samples, n_modes)
        分层采样的 KL 系数矩阵
    """
    if rng is None:
        rng = np.random.default_rng(42)

    from scipy.special import ndtri  # 正态逆 CDF

    xi = np.zeros((n_samples, n_modes))
    for k in range(n_modes):
        # 拉丁超立方: 随机排列 + 层内均匀
        perm = rng.permutation(n_samples)
        u = rng.uniform(0.0, 1.0, size=n_samples)
        p = (perm + u) / n_samples
        p = np.clip(p, 1.0e-10, 1.0 - 1.0e-10)
        xi[:, k] = ndtri(p)

    return xi


def compute_empirical_pdf(data, n_bins=50):
    """从数据计算经验 PDF (用于自适应分层).

    参数
    ----
    data : ndarray
    n_bins : int

    返回
    ----
    pdf : ndarray, shape (n_bins,)
    bin_centers : ndarray, shape (n_bins,)
    """
    counts, edges = np.histogram(data, bins=n_bins, density=True)
    pdf = counts * np.diff(edges)
    pdf = pdf / np.sum(pdf)  # 归一化
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return pdf, bin_centers
