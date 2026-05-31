"""
构型空间采样与概率分布模块
============================
基于种子项目:
  - 1079_simplex_grid     : 单形网格生成与组合排序
  - 541_histogram_pdf_sample : 直方图PDF/CDF采样（逆变换采样）

核心数学模型:
  1. M维单形网格点:
     g_1 + g_2 + ... + g_{M+1} = N,  g_i ≥ 0
     总点数:  C(N+M, M) = Π_{i=1}^{M} (N+i)/i
     网格点到笛卡尔坐标映射:
       x = (1/N) * Σ_{j=1}^{M+1} g_j * v_j

  2. 直方图近似与逆变换采样:
     (a) 将连续PDF f(x) 在 bins 上离散为直方图: b_p[i] = f(mid_i)
     (b) 归一化:  b_p ← b_p / Σ b_p·Δx
     (c) 构造CDF:  c_y[k] = Σ_{i<k} b_p[i]·Δx_i
     (d) 采样:  生成 r∼U(0,1)，二分查找区间 [c_x[left], c_x[left+1]]，
         线性插值:  x = c_x[left] + (r - c_y[left]) / (c_y[left+1] - c_y[left]) * Δx

  3. 构型空间受限采样:
     将归一化关节坐标映射到单形后，在单形内均匀或按PDF加权采样，
     再反归一化回实际关节范围。
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


# ---------------------------------------------------------------------------
# 单形网格（1079_simplex_grid）
# ---------------------------------------------------------------------------

def simplex_grid_size(m: int, n: int) -> int:
    r"""
    M维单形、N阶网格的格点总数:
      ng = C(N+M, M) = Π_{i=1}^{M} (N+i)/i
    """
    if m < 0 or n < 0:
        return 0
    ng = 1
    for i in range(1, m + 1):
        ng = ng * (n + i) // i
    return ng


def comp_next_grlex(kc: int, xc: np.ndarray) -> Optional[np.ndarray]:
    r"""
    生成下一个 graded lexicographic 序的组合。
    xc: 长度为 kc 的非负整数数组，满足 sum(xc) = N（某常数）。
    返回下一个组合，或 None 若已到最后一个。
    """
    xc = np.asarray(xc, dtype=int)
    if xc.size == 0:
        return None
    # 找到第一个可增大的位置
    for i in range(xc.size - 2, -1, -1):
        if xc[i] > 0:
            xc[i] -= 1
            xc[i + 1] += 1
            t = np.sum(xc[i + 2:])
            xc[-1] += t
            if i + 2 < xc.size:
                xc[i + 2:-1] = 0
            return xc.copy()
    return None


def simplex_grid_index_all(m: int, n: int) -> np.ndarray:
    r"""
    生成M维单形、N阶的所有网格索引（组合表示）。
    返回 (ng, m+1) 的整数数组。
    """
    ng = simplex_grid_size(m, n)
    g = np.zeros((ng, m + 1), dtype=int)
    g[0, 0] = n
    for idx in range(1, ng):
        prev = g[idx - 1].copy()
        nxt = comp_next_grlex(m + 1, prev)
        if nxt is None:
            break
        g[idx] = nxt
    return g


def simplex_grid_index_to_point(m: int, n: int, g: np.ndarray,
                                 vertices: np.ndarray) -> np.ndarray:
    r"""
    将单形网格索引映射到笛卡尔坐标点。
    vertices: (m+1, d) 单形顶点。
    映射公式:
      x = (1/n) * Σ_{j=0}^{m} g_j * v_j
    """
    g = np.asarray(g, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    if n == 0:
        return vertices[0].copy()
    return (g @ vertices) / float(n)


# ---------------------------------------------------------------------------
# 直方图PDF/CDF采样（541_histogram_pdf_sample）
# ---------------------------------------------------------------------------

def pdf_to_histogram(pdf_func: Callable, n_bins: int,
                     a: float = -1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    在区间 [a,b] 上将连续PDF离散为直方图。
    返回:
      b_l : bin 左边界
      b_r : bin 右边界
      b_p : 各bin的PDF中点值（已归一化使总面积为1）
    """
    edges = np.linspace(a, b, n_bins + 1)
    b_l = edges[:-1]
    b_r = edges[1:]
    mids = 0.5 * (b_l + b_r)
    b_p = np.array([pdf_func(m) for m in mids], dtype=float)
    # 防止负值或零
    b_p = np.maximum(b_p, 0.0)
    widths = b_r - b_l
    total = np.sum(b_p * widths)
    if total > 1e-14:
        b_p = b_p / total
    else:
        b_p = np.ones(n_bins) / (widths.sum())
    return b_l, b_r, b_p


def histogram_to_cdf(b_l: np.ndarray, b_r: np.ndarray,
                     b_p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    由直方图构造离散CDF表。
    c_x: CDF的自变量采样点（包含右端点）
    c_y: 对应的累积概率值
    """
    widths = b_r - b_l
    masses = b_p * widths
    c_y = np.zeros(b_p.size + 1, dtype=float)
    c_y[1:] = np.cumsum(masses)
    c_x = np.zeros(b_p.size + 1, dtype=float)
    c_x[0] = b_l[0]
    c_x[1:] = b_r
    # 确保最后一个值为1
    if c_y[-1] < 1e-14:
        c_y[-1] = 1.0
    else:
        c_y = c_y / c_y[-1]
    return c_x, c_y


def cdf_sample(c_x: np.ndarray, c_y: np.ndarray, n_samples: int,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    r"""
    逆变换采样：由CDF生成 n_samples 个样本。
    对每个均匀随机数 r∈[0,1]，二分查找所在区间后线性插值。
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    rvals = rng.random(n_samples)
    samples = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        r = rvals[i]
        # 二分查找：找到最大的 left 使得 c_y[left] ≤ r
        left = 0
        right = c_y.size - 1
        while left < right:
            mid = (left + right) // 2
            if c_y[mid] <= r:
                left = mid
            else:
                right = mid
            if right - left <= 1:
                break
        # 确保不越界
        if left >= c_y.size - 1:
            left = c_y.size - 2
        if c_y[left + 1] - c_y[left] < 1e-14:
            samples[i] = c_x[left]
        else:
            # 线性插值
            t = (r - c_y[left]) / (c_y[left + 1] - c_y[left])
            t = np.clip(t, 0.0, 1.0)
            samples[i] = c_x[left] + t * (c_x[left + 1] - c_x[left])
    return samples


# ---------------------------------------------------------------------------
# 构型空间采样器
# ---------------------------------------------------------------------------

class ConfigurationSampler:
    r"""
    机械臂构型空间（7维关节空间）采样器。
    结合单形网格和概率密度采样，在约束空间内生成合法构型。
    """

    def __init__(self, q_min: np.ndarray, q_max: np.ndarray,
                 n_dof: int = 7, seed: int = 42):
        self.q_min = np.asarray(q_min, dtype=float).reshape(-1)
        self.q_max = np.asarray(q_max, dtype=float).reshape(-1)
        self.n_dof = self.q_min.size
        self.rng = np.random.default_rng(seed=seed)

    def uniform_random(self, n_samples: int) -> np.ndarray:
        """在关节限位内均匀随机采样。"""
        return self.rng.uniform(self.q_min, self.q_max, size=(n_samples, self.n_dof))

    def simplex_grid_sample(self, n_per_dim: int = 4) -> np.ndarray:
        r"""
        将归一化关节空间 [-1,1]^n_dof 映射到高维单形上进行规则网格采样。
        实际做法：对每个维度独立生成n_per_dim个采样点，然后做笛卡尔积后筛选合法点。
        这里展示单形映射：将每个构型 q 映射为单形坐标
          s_i = (q_i - q_min[i]) / (q_max[i] - q_min[i])   ∈ [0,1]
        然后将其嵌入到 n_dof+1 维单形上（通过引入一个松弛变量使和为常数）。
        """
        # 生成各维度的均匀网格点
        grids = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(self.n_dof)]
        # 使用笛卡尔积（为了计算效率，限制总点数不超过5000）
        from itertools import product
        total = n_per_dim ** self.n_dof
        if total > 5000:
            # 降采样：随机采样5000个组合
            samples = self.rng.random(size=(5000, self.n_dof))
        else:
            samples = np.array(list(product(*grids)), dtype=float)
        # 映射回实际关节范围
        q_samples = self.q_min + samples * (self.q_max - self.q_min)
        return q_samples

    def pdf_weighted_sample(self, pdf_func: Callable, n_bins: int,
                            n_samples: int, dim: int = 0) -> np.ndarray:
        r"""
        在指定维度上按PDF加权采样，其余维度均匀随机。
        用于在障碍物密集的方向上增加采样密度。
        """
        dim = int(dim) % self.n_dof
        b_l, b_r, b_p = pdf_to_histogram(
            pdf_func, n_bins,
            a=float(self.q_min[dim]), b=float(self.q_max[dim])
        )
        c_x, c_y = histogram_to_cdf(b_l, b_r, b_p)
        weighted = cdf_sample(c_x, c_y, n_samples, rng=self.rng)
        others = self.rng.uniform(self.q_min, self.q_max, size=(n_samples, self.n_dof))
        others[:, dim] = weighted
        return others

    def gaussian_mixture_sample(self, n_samples: int,
                                 means: List[np.ndarray],
                                 covs: List[np.ndarray],
                                 weights: Optional[np.ndarray] = None) -> np.ndarray:
        r"""
        高斯混合模型采样：在关节空间中按多个高斯分布的混合采样。
        用于在已知狭窄通道附近集中采样。
        """
        if weights is None:
            weights = np.ones(len(means)) / len(means)
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        samples = []
        for _ in range(n_samples):
            k = self.rng.choice(len(means), p=weights)
            mean = np.asarray(means[k])
            cov = np.asarray(covs[k])
            # 确保协方差正定
            cov = cov + 1e-4 * np.eye(self.n_dof)
            s = self.rng.multivariate_normal(mean, cov)
            s = np.clip(s, self.q_min, self.q_max)
            samples.append(s)
        return np.array(samples, dtype=float)
