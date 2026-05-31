"""
cvt_sampler.py
==============
基于种子项目 248_cvt_2d_sampling 的 Centroidal Voronoi Tessellation 采样模块。
在物理信息 GAN 中提供最优采样点布局，用于：
  1. 物理域上的高精度残差评估点放置；
  2. 隐空间中的结构化采样，提升生成多样性。

核心数学：
  Lloyd 迭代算法：
    给定生成元集合 G = {g_1, ..., g_k}，重复以下步骤直至收敛：
      (1) 对域 Ω 进行 Voronoi 划分：
            V_i = { x ∈ Ω | ||x - g_i|| ≤ ||x - g_j||, ∀j ≠ i }
      (2) 更新生成元为对应 Voronoi 单元的质心：
            g_i^{new} = ∫_{V_i} x·ρ(x) dx / ∫_{V_i} ρ(x) dx
      其中 ρ(x) 为密度函数（均匀密度时 ρ ≡ 1）。

  CVT 能量泛函：
      E(G) = Σ_{i=1}^k ∫_{V_i} ||x - g_i||² · ρ(x) dx
    Lloyd 迭代单调递减 E(G)，收敛到局部极小值。

  采样域内最近邻搜索使用 Delaunay 三角剖分加速。
"""

import numpy as np
from scipy.spatial import Delaunay, cKDTree


def cvt_energy(generators: np.ndarray, samples: np.ndarray,
               weights: np.ndarray = None) -> float:
    """
    计算 CVT 能量泛函 E(G)。

    Parameters
    ----------
    generators : np.ndarray, shape (k, dim)
        生成元位置。
    samples : np.ndarray, shape (m, dim)
        样本点。
    weights : np.ndarray, shape (m,), optional
        样本权重（默认为均匀权重 1/m）。

    Returns
    -------
    energy : float
        CVT 能量。
    """
    if weights is None:
        weights = np.ones(samples.shape[0]) / samples.shape[0]
    tree = cKDTree(generators)
    dists, _ = tree.query(samples, k=1)
    energy = float(np.sum(weights * dists ** 2))
    return energy


def lloyd_step(generators: np.ndarray, samples: np.ndarray,
               weights: np.ndarray = None) -> np.ndarray:
    """
    执行一次 Lloyd 迭代：将生成元更新为 Voronoi 单元的样本质心。

    Parameters
    ----------
    generators : np.ndarray, shape (k, dim)
        当前生成元。
    samples : np.ndarray, shape (m, dim)
        样本点。
    weights : np.ndarray, shape (m,), optional
        样本权重。

    Returns
    -------
    new_generators : np.ndarray, shape (k, dim)
        更新后的生成元。
    """
    if weights is None:
        weights = np.ones(samples.shape[0]) / samples.shape[0]
    tree = cKDTree(generators)
    _, idx = tree.query(samples, k=1)
    k = generators.shape[0]
    new_gens = np.zeros_like(generators)
    for i in range(k):
        mask = idx == i
        if np.sum(mask) == 0:
            # 空单元：保持原生成元
            new_gens[i] = generators[i]
        else:
            w = weights[mask]
            new_gens[i] = np.sum(samples[mask] * w[:, None], axis=0) / np.sum(w)
    return new_gens


def cvt_2d_sampling(k: int = 16, n_samples: int = 5000,
                    itermax: int = 50, tol: float = 1e-5,
                    bounds: tuple = ((0.0, 1.0), (0.0, 1.0)),
                    seed: int = None) -> np.ndarray:
    """
    在二维矩形域内计算 Centroidal Voronoi Tessellation 采样点。

    Parameters
    ----------
    k : int
        生成元数量。
    n_samples : int
        每次 Lloyd 迭代使用的样本点数。
    itermax : int
        最大迭代次数。
    tol : float
        生成元位移收敛阈值。
    bounds : tuple
        ((xmin, xmax), (ymin, ymax)) 定义矩形域。
    seed : int, optional
        随机种子。

    Returns
    -------
    generators : np.ndarray, shape (k, 2)
        CVT 最优采样点。
    """
    rng = np.random.default_rng(seed)
    # 初始化生成元
    gens = rng.random((k, 2))
    gens[:, 0] = gens[:, 0] * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
    gens[:, 1] = gens[:, 1] * (bounds[1][1] - bounds[1][0]) + bounds[1][0]

    for it in range(itermax):
        # 生成均匀样本
        samples = rng.random((n_samples, 2))
        samples[:, 0] = samples[:, 0] * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
        samples[:, 1] = samples[:, 1] * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
        new_gens = lloyd_step(gens, samples)
        motion = float(np.max(np.sqrt(np.sum((new_gens - gens) ** 2, axis=1))))
        gens = new_gens
        if motion < tol:
            break
    return gens


def cvt_latent_samples(k: int = 32, dim: int = 8, n_samples: int = 10000,
                       itermax: int = 30, seed: int = None) -> np.ndarray:
    """
    在高维隐空间 [0,1]^dim 内计算 CVT 结构化采样点，
    用于 GAN 隐空间的多样化覆盖。

    Parameters
    ----------
    k : int
        采样点数量。
    dim : int
        隐空间维度。
    n_samples : int
        Lloyd 迭代样本数。
    itermax : int
        最大迭代次数。
    seed : int, optional
        随机种子。

    Returns
    -------
    generators : np.ndarray, shape (k, dim)
        隐空间 CVT 采样点。
    """
    rng = np.random.default_rng(seed)
    gens = rng.random((k, dim))
    for it in range(itermax):
        samples = rng.random((n_samples, dim))
        new_gens = lloyd_step(gens, samples)
        motion = float(np.max(np.sqrt(np.sum((new_gens - gens) ** 2, axis=1))))
        gens = new_gens
        if motion < 1e-4:
            break
    return gens
