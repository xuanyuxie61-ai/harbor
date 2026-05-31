"""
拉丁超立方采样模块
==================
基于种子项目 652_latin_random 的核心算法改造。

在金融工程的蒙特卡洛模拟中，拉丁超立方采样(LHS)作为方差缩减技术，
相比纯随机采样能更均匀地覆盖输入空间，显著提高路径模拟的收敛速度。

数学原理:
---------
对于 d 维空间、N 个采样点，LHS 确保每一维上恰好有一个点落入区间:
    [ (j-1)/N , j/N ),  j = 1, ..., N

构造方法:
    1. 对每一维 i，生成 1..N 的随机排列 perm_i
    2. 对每个样本点 j，第 i 维坐标为:
        X_{i,j} = (perm_i[j] - 1 + U_{i,j}) / N
       其中 U_{i,j} ~ Uniform(0,1)

在金融随机波动率模型中，LHS用于生成标准正态随机变量驱动资产价格
与波动率过程的联合演化，有效降低蒙特卡洛估计方差。
"""

import numpy as np
from math import sqrt, log


class LatinHypercubeSampler:
    """
    拉丁超立方采样器，支持标准正态变换与相关系数矩阵嵌入。
    """

    def __init__(self, dim_num, point_num, seed=None):
        """
        参数:
        ------
        dim_num   : int, 空间维度
        point_num : int, 采样点数量
        seed      : int, 随机种子（可选）
        """
        if dim_num <= 0 or point_num <= 0:
            raise ValueError("dim_num和point_num必须为正整数")
        self.dim_num = dim_num
        self.point_num = point_num
        self.rng = np.random.default_rng(seed)

    def sample_uniform(self):
        """
        生成 [0,1]^d 上的拉丁超立方样本。

        返回:
        ------
        ndarray, 形状 (dim_num, point_num)
        """
        x = self.rng.random((self.dim_num, self.point_num))
        for i in range(self.dim_num):
            perm = self.rng.permutation(self.point_num)
            for j in range(self.point_num):
                x[i, j] = (perm[j] + x[i, j]) / self.point_num
        return x

    def sample_normal(self, mean=None, cov=None):
        """
        生成服从多元正态分布 N(mean, cov) 的拉丁超立方样本。

        使用逆变换采样:
            Z = Φ^{-1}(U),  U为LHS均匀样本
        然后通过Cholesky分解嵌入相关性:
            X = mean + L · Z,  cov = L L^T

        参数:
        ------
        mean : array, 均值向量, 默认零向量
        cov  : array, 协方差矩阵, 默认单位阵

        返回:
        ------
        ndarray, 形状 (dim_num, point_num)
        """
        # 先生成均匀LHS样本
        u = self.sample_uniform()
        # 逆正态变换（避免极端尾部，截断到[1e-10, 1-1e-10]）
        eps = 1e-10
        u = np.clip(u, eps, 1.0 - eps)
        z = np.zeros_like(u)
        for i in range(self.dim_num):
            for j in range(self.point_num):
                z[i, j] = _inverse_cdf_normal(u[i, j])

        # 嵌入相关性
        if cov is not None:
            cov = np.asarray(cov, dtype=np.float64)
            if cov.shape != (self.dim_num, self.dim_num):
                raise ValueError(f"协方差矩阵维度不匹配")
            # Cholesky分解
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                # 若cov不正定，进行特征值修正
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            z = L @ z

        if mean is not None:
            mean = np.asarray(mean, dtype=np.float64).reshape(self.dim_num, 1)
            z = z + mean

        return z

    def sample_for_heston(self, rho, point_num=None):
        """
        专门为Heston模型生成具有相关性ρ的二维标准正态样本。

        Heston模型中需要两个相关的布朗运动:
            dW^S · dW^v = ρ dt

        协方差矩阵:
            Σ = [[1, ρ],
                 [ρ, 1]]

        返回:
        ------
        ndarray, 形状 (2, point_num)，第一行为dW^S，第二行为dW^v
        """
        if point_num is None:
            point_num = self.point_num
        old_point_num = self.point_num
        self.point_num = point_num
        cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
        samples = self.sample_normal(cov=cov)
        self.point_num = old_point_num
        return samples


def _inverse_cdf_normal(p):
    """
    使用有理近似计算标准正态逆累积分布函数 Φ^{-1}(p)。
    基于Peter J. Acklam近似公式，精度达1e-9。
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p必须在(0,1)开区间内")

    # 分情况处理
    if p < 0.5:
        q = p
        sign = -1.0
    else:
        q = 1.0 - p
        sign = 1.0

    if q < 1e-300:
        # 极端尾部使用渐近展开
        t = sqrt(-2.0 * log(q))
        return sign * (t - (log(t) + 0.5 * log(4.0 * np.pi)) / t)

    # Acklam近似系数
    a1 = -3.969683028665376e+01
    a2 =  2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 =  1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 =  2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 =  1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 =  6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 =  4.374664141464968e+00
    c6 =  2.938163982698783e+00

    d1 =  7.784695709041462e-03
    d2 =  3.224671290700398e-01
    d3 =  2.445134137142996e+00
    d4 =  3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # 尾部有理近似
        q_sqrt = sqrt(-2.0 * log(q))
        x = (((((c1 * q_sqrt + c2) * q_sqrt + c3) * q_sqrt + c4) * q_sqrt + c5) * q_sqrt + c6) \
            / ((((d1 * q_sqrt + d2) * q_sqrt + d3) * q_sqrt + d4) * q_sqrt + 1.0)
    elif p <= p_high:
        # 中心有理近似
        r = q - 0.5
        r2 = r * r
        x = (((((a1 * r2 + a2) * r2 + a3) * r2 + a4) * r2 + a5) * r2 + a6) * r \
            / (((((b1 * r2 + b2) * r2 + b3) * r2 + b4) * r2 + b5) * r2 + 1.0)
    else:
        # 上尾部
        q_sqrt = sqrt(-2.0 * log(q))
        x = -(((((c1 * q_sqrt + c2) * q_sqrt + c3) * q_sqrt + c4) * q_sqrt + c5) * q_sqrt + c6) \
            / ((((d1 * q_sqrt + d2) * q_sqrt + d3) * q_sqrt + d4) * q_sqrt + 1.0)

    # 使用Newton-Raphson精化一步
    # Φ(x) 的误差
    e = 0.5 * (1.0 + np.sign(x) * np.sqrt(1.0 - np.exp(-2.0/np.pi * x * x))) - p
    # 这里简化为直接返回，因为Acklam近似已经足够精确
    return sign * abs(x)
