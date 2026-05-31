"""
statistical_covariance.py
气溶胶观测统计协方差分析模块

整合原项目:
  - 029_asa053: Wishart 随机矩阵生成

功能:
  1. 生成气溶胶光学厚度 (AOD) 观测的样本协方差矩阵
  2. 使用 Wishart 分布对协方差矩阵进行统计建模
  3. 特征值谱分析，识别主模态

核心公式:
  - Wishart 分布 W_p(n, Σ):
      若 X_i ~ N_p(0, Σ) 独立，则 S = Σ_{i=1}^n X_i X_i^T ~ W_p(n, Σ)
    概率密度:
      f(S) = |S|^{(n-p-1)/2} exp(-tr(Σ^{-1}S)/2) / ( 2^{np/2} |Σ|^{n/2} Γ_p(n/2) )

  - 多元正态采样:
      X = Σ^{1/2} Z,  Z ~ N(0, I)

  - 特征值分解 (EOF 分析):
      Σ = V Λ V^T
      主成分: PC_k(t) = V_k^T x(t)
"""

import numpy as np
from math import sqrt, pi, exp, lgamma
from numerical_utils import rnorm, wilson_hilferty_chi_square


class StatisticalError(Exception):
    pass


def cholesky_factor(Sigma):
    """
    计算协方差矩阵 Σ 的 Cholesky 分解 D (上三角)。
    返回列优先存储的上三角元素向量。
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    p = Sigma.shape[0]
    if Sigma.shape[0] != Sigma.shape[1]:
        raise StatisticalError("cholesky_factor: 输入必须为方阵")

    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        # 若不正定，添加小量
        Sigma = Sigma + np.eye(p) * 1e-8
        L = np.linalg.cholesky(Sigma)

    # 提取上三角 (列优先)
    d = []
    for j in range(p):
        for i in range(j + 1):
            d.append(L[i, j])
    return np.array(d, dtype=np.float64)


def wishart_variate(Sigma, n, np_var=None):
    """
    生成 Wishart 分布 W_p(n, Σ) 的随机矩阵。

    基于 asa053 (Smith & Hocking, 1972) 算法:
      1. 生成 p(p+1)/2 个独立标准正态变量 SB
      2. 对角线用卡方分布替代 (Wilson-Hilferty 近似)
      3. 通过 Cholesky 因子 D 变换: SA = D @ SB，然后 S = SA @ SA^T / n

    参数:
      Sigma: (p, p) 尺度矩阵 (协方差)
      n: 自由度
      np_var: 变量维度 p，若 None 则自动推断

    返回:
      S: (p, p) 样本协方差矩阵
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if np_var is None:
        np_var = Sigma.shape[0]
    p = np_var

    if n < 1 or n > p:
        raise StatisticalError(f"wishart_variate: 自由度 n={n} 必须在 [1, p={p}]")

    d = cholesky_factor(Sigma)
    nnp = p * (p + 1) // 2

    # 生成独立标准正态
    sb = np.zeros(nnp, dtype=np.float64)
    k = 0
    while k < nnp:
        z1, z2 = rnorm()
        sb[k] = z1
        k += 1
        if k < nnp:
            sb[k] = z2
            k += 1

    # 对角线替换为卡方变量
    sa = np.zeros(nnp, dtype=np.float64)
    ns = 0
    for i in range(1, p + 1):
        df = p - i + 1
        ns += i
        sb[ns - 1] = wilson_hilferty_chi_square(df, sb[ns - 1])

    # 矩阵乘法: SA = D @ SB (上三角)
    # 简化: 通过 numpy 重构矩阵
    D_mat = np.zeros((p, p), dtype=np.float64)
    idx = 0
    for j in range(p):
        for i in range(j + 1):
            D_mat[i, j] = d[idx]
            idx += 1

    SB_mat = np.zeros((p, p), dtype=np.float64)
    idx = 0
    for j in range(p):
        for i in range(j + 1):
            SB_mat[i, j] = sb[idx]
            idx += 1

    SA_mat = D_mat @ SB_mat
    # Wishart 样本: S = SA @ SA^T / n
    S = (SA_mat @ SA_mat.T) / n
    return S


def sample_covariance_matrix(data):
    """
    由数据矩阵计算样本协方差矩阵。

    参数:
      data: (n_samples, p) 数据矩阵

    返回:
      S: (p, p) 无偏样本协方差
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    if n < 2:
        raise StatisticalError("sample_covariance_matrix: 样本数至少为 2")
    mean = np.mean(data, axis=0)
    centered = data - mean
    S = (centered.T @ centered) / (n - 1)
    return S


def eof_analysis(cov_matrix, num_modes=None):
    """
    经验正交函数 (EOF) 分析。

    参数:
      cov_matrix: (p, p) 协方差矩阵
      num_modes: 保留模态数，None 则保留全部

    返回:
      eigenvalues: 特征值 (方差解释)
      eigenvectors: 特征向量 (EOF 空间模态)
      explained_variance_ratio: 各模态方差贡献率
    """
    cov = np.asarray(cov_matrix, dtype=np.float64)
    p = cov.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 按降序排列
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 非负化
    eigvals = np.maximum(eigvals, 0.0)
    total = np.sum(eigvals)
    if total > 0:
        evr = eigvals / total
    else:
        evr = np.zeros_like(eigvals)

    if num_modes is not None:
        eigvals = eigvals[:num_modes]
        eigvecs = eigvecs[:, :num_modes]
        evr = evr[:num_modes]

    return eigvals, eigvecs, evr


def aod_covariance_model(stations_lat_lon, correlation_length=500.0, sigma_aod=0.15):
    """
    构建基于指数衰减空间相关结构的 AOD 理论协方差矩阵。

    公式:
      Σ_ij = σ_aod^2 * exp( -d_ij / L )
    其中 d_ij 为站点间球面距离 (km)，L 为相关长度。

    参数:
      stations_lat_lon: (N, 2) 站点经纬度
      correlation_length: 相关长度 (km)
      sigma_aod: AOD 标准差

    返回:
      Sigma: (N, N) 协方差矩阵
    """
    from atmospheric_mesh import ll_degrees_to_distance_earth

    N = stations_lat_lon.shape[0]
    Sigma = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        Sigma[i, i] = sigma_aod ** 2
        for j in range(i + 1, N):
            d = ll_degrees_to_distance_earth(
                stations_lat_lon[i, 0], stations_lat_lon[i, 1],
                stations_lat_lon[j, 0], stations_lat_lon[j, 1]
            )
            val = (sigma_aod ** 2) * exp(-d / correlation_length)
            Sigma[i, j] = val
            Sigma[j, i] = val

    # 确保正定性
    eigvals = np.linalg.eigvalsh(Sigma)
    min_eig = np.min(eigvals)
    if min_eig < 1e-10:
        Sigma = Sigma + np.eye(N) * (1e-10 - min_eig)
    return Sigma
