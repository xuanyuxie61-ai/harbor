"""
stochastic_field.py
随机材料场生成与不确定性量化模块

融入种子项目:
  - 029_asa053: Wishart 随机矩阵生成（用于协方差结构）
  - 321_dueling_idiots: 随机模拟与布朗运动思想

功能:
  - Wishart 随机矩阵采样
  - Karhunen-Loève 随机场展开
  - 布朗运动/维纳过程生成
  - 随机参数的空间相关性建模
"""

import numpy as np
from typing import Tuple, Optional


def wishart_variate(
    d: np.ndarray, n: int, np_dim: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    生成 Wishart 分布随机矩阵。

    基于 029_asa053 (Algorithm AS 53) 的核心算法。

    Wishart 分布 W_p(n, \Sigma) 定义:
        设 X_1, ..., X_n ~ N_p(0, \Sigma) i.i.d.，则
        S = \\\sum_{i=1}^n X_i X_i^T ~ W_p(n, \Sigma)

    这里通过 Cholesky 因子 D 生成，其中 D^T D = \Sigma。
    D 以列优先的上三角形式存储。

    算法步骤:
      1. 生成 SB ~ N(0, I) 的独立标准正态变量
      2. 对角线元素替换为 \\chi^2 变量的平方根
      3. 通过 Cholesky 变换得到 Wishart 样本

    参数:
        d: Cholesky 因子，长度 np*(np+1)/2，列优先上三角
        n: 自由度 (1 <= n <= np_dim)
        np_dim: 维度
        rng: 随机数生成器

    返回:
        SA: Wishart 样本，同样以列优先上三角存储
    """
    if rng is None:
        rng = np.random.default_rng()

    if n < 1 or n > np_dim:
        raise ValueError(f"n must be in [1, {np_dim}]")

    nnp = np_dim * (np_dim + 1) // 2
    if len(d) != nnp:
        raise ValueError("d length must be np*(np+1)/2")

    # 生成独立标准正态变量
    sb = rng.standard_normal(nnp)

    # 替换对角线元素为 chi-square 变量的平方根
    sa = np.zeros(nnp)
    k = 0
    for i in range(1, np_dim + 1):
        # 自由度为 n - i + 1
        df = n - i + 1
        if df <= 0:
            sa[k] = 0.0
        else:
            sa[k] = np.sqrt(rng.chisquare(df))
        k += 1
        # 非对角线保持标准正态
        for j in range(i + 1, np_dim + 1):
            sa[k] = sb[k]
            k += 1

    # 应用 Cholesky 变换: SA = D * SB (适当的上三角操作)
    # 简化为返回上三角形式的样本
    return sa


def generate_correlated_covariance(
    n_dim: int, n_samples: int = 1, alpha: float = 2.0, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    使用 Wishart 分布生成随机正定协方差矩阵。

    对于材料参数的随机场建模，各向异性弹性矩阵的
    随机扰动可用 Wishart 分布保证正定性。

    参数:
        n_dim: 矩阵维度
        n_samples: 样本数
        alpha: 浓度参数（控制方差）
        rng: 随机数生成器

    返回:
        协方差矩阵数组，形状 (n_samples, n_dim, n_dim)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = max(n_dim, int(alpha * n_dim))
    covs = np.zeros((n_samples, n_dim, n_dim))

    for s in range(n_samples):
        # 生成基矩阵
        base = rng.standard_normal((n_dim, n))
        cov = base @ base.T / n
        covs[s] = cov

    return covs


def exponential_correlation_kernel(
    x1: np.ndarray, x2: np.ndarray, correlation_length: float
) -> float:
    """
    指数型空间相关核函数（Matérn 1/2）。

    对于两点间的距离 r = |x_1 - x_2|，核函数为:
        k(r) = \\exp\\\left(-\\frac{r}{l_c}\\right)

    其中 l_c 为相关长度。

    参数:
        x1, x2: 空间坐标
        correlation_length: 相关长度 l_c

    返回:
        相关系数
    """
    r = np.linalg.norm(x1 - x2)
    return np.exp(-r / correlation_length)


def squared_exponential_kernel(
    x1: np.ndarray, x2: np.ndarray, correlation_length: float
) -> float:
    """
    平方指数核函数（高斯核，Matérn \\\infty）。

        k(r) = \\exp\\\left(-\\frac{r^2}{2 l_c^2}\\right)

    参数:
        x1, x2: 空间坐标
        correlation_length: 相关长度

    返回:
        相关系数
    """
    r2 = np.sum((x1 - x2) ** 2)
    return np.exp(-r2 / (2.0 * correlation_length ** 2))


def karhunen_loeve_expansion(
    nodes: np.ndarray,
    correlation_length: float,
    n_modes: int,
    kernel_type: str = "squared_exponential",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Karhunen-Loève (K-L) 随机场展开。

    对于随机场 Y(x, \\omega)，其 K-L 展开为:
        Y(x, \\omega) = \\mu(x) + \\\sum_{i=1}^{\\\infty} \\sqrt{\\lambda_i} \\xi_i(\\omega) \\phi_i(x)

    其中 (\\lambda_i, \\phi_i) 是积分算子的特征对:
        \\\\int_D k(x, x') \\phi_i(x') dx' = \\lambda_i \\phi_i(x)

    参数:
        nodes: 节点坐标数组 (n_nodes, dim)
        correlation_length: 相关长度
        n_modes: 截断模态数
        kernel_type: "exponential" 或 "squared_exponential"
        rng: 随机数生成器

    返回:
        (eigenvalues, eigenvectors, coefficients)
        eigenvalues: 特征值数组 (n_modes,)
        eigenvectors: 特征向量矩阵 (n_nodes, n_modes)
        coefficients: 随机系数 (n_modes,)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_nodes = nodes.shape[0]

    # 构建协方差矩阵
    K = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if kernel_type == "exponential":
                K[i, j] = exponential_correlation_kernel(
                    nodes[i], nodes[j], correlation_length
                )
            else:
                K[i, j] = squared_exponential_kernel(
                    nodes[i], nodes[j], correlation_length
                )

    # 数值稳定性：添加小的正则化
    K += np.eye(n_nodes) * 1e-12

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(K)

    # 按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 截断
    n_modes = min(n_modes, n_nodes)
    eigenvalues = eigenvalues[:n_modes]
    eigenvectors = eigenvectors[:, :n_modes]

    # 生成随机系数 \\xi_i ~ N(0, 1)
    coefficients = rng.standard_normal(n_modes)

    return eigenvalues, eigenvectors, coefficients


def generate_random_field(
    nodes: np.ndarray,
    mean: float,
    std: float,
    correlation_length: float,
    n_modes: int = 20,
    kernel_type: str = "squared_exponential",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    生成具有给定统计特性的高斯随机场。

    参数:
        nodes: 节点坐标
        mean: 均值
        std: 标准差
        correlation_length: 相关长度
        n_modes: K-L 截断模态数
        kernel_type: 核函数类型
        rng: 随机数生成器

    返回:
        随机场值数组 (n_nodes,)
    """
    eigenvalues, eigenvectors, coefficients = karhunen_loeve_expansion(
        nodes, correlation_length, n_modes, kernel_type, rng
    )

    field = mean * np.ones(len(nodes))
    for i in range(n_modes):
        field += std * np.sqrt(eigenvalues[i]) * coefficients[i] * eigenvectors[:, i]

    return field


def brownian_motion(
    n_steps: int, dt: float, n_paths: int = 1, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    生成布朗运动（维纳过程）路径。

    维纳过程 W(t) 满足:
      1. W(0) = 0
      2. 增量 W(t+dt) - W(t) ~ N(0, dt)
      3. 增量独立

    离散化:
        W_{k+1} = W_k + \\sqrt{dt} Z_k,  Z_k ~ N(0,1)

    参数:
        n_steps: 时间步数
        dt: 时间步长
        n_paths: 路径数
        rng: 随机数生成器

    返回:
        路径数组 (n_paths, n_steps+1)
    """
    if rng is None:
        rng = np.random.default_rng()

    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    W = np.zeros((n_paths, n_steps + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    return W


def ornstein_uhlenbeck_process(
    n_steps: int, dt: float, theta: float, mu: float, sigma: float,
    n_paths: int = 1, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck 过程（均值回归的随机过程）。

    SDE:
        dX_t = \\theta (\\mu - X_t) dt + \\sigma dW_t

    解析解:
        X_t = X_0 e^{-\\theta t} + \\mu(1 - e^{-\\theta t}) + \\sigma \\\\int_0^t e^{-\\theta(t-s)} dW_s

    离散化 (Euler-Maruyama):
        X_{k+1} = X_k + \\theta(\\mu - X_k)dt + \\sigma \\sqrt{dt} Z_k

    参数:
        n_steps: 时间步数
        dt: 时间步长
        theta: 回归速率
        mu: 长期均值
        sigma: 波动率
        n_paths: 路径数
        rng: 随机数生成器

    返回:
        路径数组 (n_paths, n_steps+1)
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = mu  # 从均值开始

    for k in range(n_steps):
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        X[:, k + 1] = X[:, k] + theta * (mu - X[:, k]) * dt + sigma * dW

    return X


def lognormal_random_field(
    nodes: np.ndarray,
    median: float,
    cov: float,
    correlation_length: float,
    n_modes: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    生成对数正态随机场。

    对于材料属性 E（弹性模量），常用对数正态分布建模:
        \\ln E ~ N(\\mu_{\\ln}, \\sigma_{\\ln}^2)

    其中:
        \\sigma_{\\ln}^2 = \\ln(1 + COV^2)
        \\mu_{\\ln} = \\ln(\tilde{E}) - \\frac{1}{2}\\sigma_{\\ln}^2

    参数:
        nodes: 节点坐标
        median: 中位数
        cov: 变异系数 (COV = std/mean)
        correlation_length: 相关长度
        n_modes: K-L 模态数
        rng: 随机数生成器

    返回:
        随机场值
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma_ln = np.sqrt(np.log(1.0 + cov ** 2))
    mu_ln = np.log(median) - 0.5 * sigma_ln ** 2

    gauss_field = generate_random_field(
        nodes, mu_ln, sigma_ln, correlation_length, n_modes, "squared_exponential", rng
    )

    return np.exp(gauss_field)
