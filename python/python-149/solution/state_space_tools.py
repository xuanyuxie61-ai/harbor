"""
state_space_tools.py
状态空间离散化、编码与数据转换工具

融合种子项目:
  - 258_cvt_metric: 变度量空间中的质心Voronoi剖分
  - 774_monoalphabetic: 单字母置换编码 → 状态空间标签编码
  - 1199_tec_to_vtk: 网格数据格式转换思想

科学背景:
  在连续状态空间 X ⊂ R^d 中，为强化学习的值函数近似，
  需要对状态空间进行高效离散化。CVT (Centroidal Voronoi Tessellation)
  提供最优的量化器:

      z_i = argmin_{z∈Ω} ∫_{V_i(z)} ρ(x) ||x - z||_A^2 dx

  其中 ||x - z||_A^2 = (x-z)^T A(x) (x-z) 为变度量距离，
  A(x) 为局部度量张量（如Fisher信息矩阵或Hessian）。

  离散化后，每个状态 x 被映射到最近生成元的索引:
      idx(x) = argmin_i d_A(x, z_i)

  该索引进一步通过置换编码进行混淆/压缩存储。
"""

import numpy as np
from typing import Tuple, Optional, Callable, List


def metric_tensor(x: np.ndarray, metric_type: str = "fisher") -> np.ndarray:
    """
    计算状态点 x 处的局部度量张量 A(x)。

    类型:
    - "euclidean": A = I
    - "fisher": A = diag(1/(x+ε))  (Fisher信息近似)
    - "anisotropic": A = diag([1, 10]) (各向异性)

    Parameters
    ----------
    x : ndarray
    metric_type : str

    Returns
    -------
    A : ndarray, shape (d, d)
        对称正定矩阵
    """
    d = len(np.atleast_1d(x))
    x_safe = np.clip(np.abs(np.atleast_1d(x)), 1e-6, 1e6)

    if metric_type == "euclidean":
        return np.eye(d)
    elif metric_type == "fisher":
        return np.diag(1.0 / x_safe)
    elif metric_type == "anisotropic":
        A = np.eye(d)
        if d >= 2:
            A[0, 0] = 1.0
            A[1, 1] = 10.0
        return A
    else:
        return np.eye(d)


def cvt_lloyd_iterate(
    generators: np.ndarray,
    n_samples: int = 5000,
    n_iter: int = 10,
    metric_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    变度量CVT Lloyd迭代。

    算法:
        1. 在区域内均匀采样 {x_s}
        2. 对每个样本，找到最近生成元（变度量距离）
        3. 将生成元更新为Voronoi区域的质心
        4. 重复

    变度量距离:
        d_A(x, z)^2 = (x-z)^T A((x+z)/2) (x-z)

    Parameters
    ----------
    generators : ndarray, shape (n, d)
        初始生成元位置
    n_samples : int
        每步采样点数
    n_iter : int
        Lloyd迭代次数
    metric_fn : callable or None
        度量张量函数 A(x)
    bounds : (xmin, xmax) or None
        区域边界
    rng : np.random.Generator or None

    Returns
    -------
    generators : ndarray, shape (n, d)
        优化后的生成元
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    if metric_fn is None:
        metric_fn = lambda x: np.eye(len(np.atleast_1d(x)))

    n, d = generators.shape

    if bounds is None:
        xmin = np.zeros(d)
        xmax = np.ones(d)
    else:
        xmin, xmax = bounds

    for it in range(n_iter):
        # 采样
        samples = rng.uniform(0, 1, (n_samples, d))
        for dd in range(d):
            samples[:, dd] = xmin[dd] + samples[:, dd] * (xmax[dd] - xmin[dd])

        # 最近生成元分配
        labels = np.zeros(n_samples, dtype=int)
        for s in range(n_samples):
            best_dist = np.inf
            best_g = 0
            xs = samples[s, :]
            for g in range(n):
                zg = generators[g, :]
                mid = 0.5 * (xs + zg)
                A = metric_fn(mid)
                diff = xs - zg
                # 确保A正定
                try:
                    dist2 = diff @ A @ diff
                except Exception:
                    dist2 = np.sum(diff ** 2)
                if dist2 < best_dist:
                    best_dist = dist2
                    best_g = g
            labels[s] = best_g

        # 更新质心
        new_generators = np.zeros_like(generators)
        counts = np.zeros(n)
        for s in range(n_samples):
            g = labels[s]
            new_generators[g, :] += samples[s, :]
            counts[g] += 1

        for g in range(n):
            if counts[g] > 0:
                new_generators[g, :] /= counts[g]
            else:
                # 若某区域无样本，随机重置
                new_generators[g, :] = xmin + rng.uniform(0, 1, d) * (xmax - xmin)

        generators = new_generators.copy()

    return generators


def state_to_index(
    x: np.ndarray,
    generators: np.ndarray,
    metric_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> int:
    """
    将连续状态映射到离散生成元索引:

        idx(x) = argmin_i ||x - z_i||_{A}

    Parameters
    ----------
    x : ndarray
    generators : ndarray, shape (n, d)

    Returns
    -------
    idx : int
        最近生成元索引
    """
    if metric_fn is None:
        metric_fn = lambda x: np.eye(len(np.atleast_1d(x)))

    n = generators.shape[0]
    best_dist = np.inf
    best_idx = 0

    for i in range(n):
        diff = x - generators[i, :]
        mid = 0.5 * (x + generators[i, :])
        A = metric_fn(mid)
        try:
            dist2 = float(diff @ A @ diff)
        except Exception:
            dist2 = float(np.sum(diff ** 2))
        if dist2 < best_dist:
            best_dist = dist2
            best_idx = i

    return best_idx


# ============================================================================
# 单字母置换编码（融合monoalphabetic）
# ============================================================================

class StateEncoder:
    """
    状态索引的置换编码器。

    将离散状态索引通过单字母置换映射为编码标签，
    用于Q-table的紧凑存储与索引混淆。

    编码映射:
        code[i] = alphabet[perm[i]]

    解码映射:
        对加密字符 c，找到 j 使得 code[j] = c，则原始索引为 j。
    """

    def __init__(self, n_states: int, rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng(seed=42)
        self.n_states = n_states
        # 生成置换
        self.perm = rng.permutation(n_states)
        self.inv_perm = np.argsort(self.perm)

    def encode(self, idx: int) -> int:
        """编码状态索引。"""
        if not (0 <= idx < self.n_states):
            idx = idx % self.n_states
        return int(self.perm[idx])

    def decode(self, code_idx: int) -> int:
        """解码状态索引。"""
        if not (0 <= code_idx < self.n_states):
            code_idx = code_idx % self.n_states
        return int(self.inv_perm[code_idx])


# ============================================================================
# 数据序列化/反序列化（融合tec_to_vtk思想）
# ============================================================================

def serialize_state_trajectory(
    t: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    metadata: Optional[dict] = None,
) -> dict:
    """
    将状态轨迹序列化为结构化字典。

    Parameters
    ----------
    t : ndarray, shape (N,)
        时间网格
    y : ndarray, shape (N, d)
        状态轨迹
    u : ndarray, shape (N,)
        控制输入序列
    metadata : dict or None
        附加元数据

    Returns
    -------
    data : dict
        结构化数据包
    """
    data = {
        "version": "1.0",
        "n_steps": len(t),
        "dim": y.shape[1] if y.ndim > 1 else 1,
        "time": t.tolist(),
        "state": y.tolist(),
        "control": u.tolist(),
    }
    if metadata is not None:
        data["metadata"] = metadata
    return data


def deserialize_state_trajectory(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    反序列化状态轨迹。

    Returns
    -------
    t, y, u : ndarrays
    """
    t = np.array(data["time"])
    y = np.array(data["state"])
    u = np.array(data["control"])
    return t, y, u
