"""
fem_sampler.py
==============
基于有限元方法的量子期望值采样与态空间插值

原项目映射:
- 398_fem1d_sample: 一维有限元采样与分段线性插值
- 418_fem3d_project: 三维有限元投影、四面体基础函数、体积计算

科学功能:
本模块将量子态的Born概率分布视为定义在离散化参数空间上的
有限元函数，利用1D/3D有限元技术进行期望值估计和概率密度插值。
这允许在粗采样网格上训练代理模型，再在密网格上精确估计
VQE能量期望值，大幅减少量子电路调用次数。
"""

import numpy as np
from typing import Tuple, Optional, List


def bracket4(t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    为每个采样值s找到其在有序数组t中的包围区间左端点索引。
    对应 398_fem1d_sample/r8vec_bracket4 的二分搜索逻辑。

    返回:
        left: 长度与s相同的数组，left[i] 满足 t[left[i]] <= s[i] <= t[left[i]+1]
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    nt = t.shape[0]
    if nt < 2:
        raise ValueError("t必须至少包含2个元素")
    left = np.zeros(s.shape[0], dtype=int)
    for idx, val in enumerate(s):
        # 二分搜索
        lo, hi = 0, nt - 2
        if val <= t[0]:
            left[idx] = 0
            continue
        if val >= t[-1]:
            left[idx] = nt - 2
            continue
        while lo < hi:
            mid = (lo + hi) // 2
            if t[mid] <= val:
                if val <= t[mid + 1]:
                    lo = mid
                    break
                lo = mid + 1
            else:
                hi = mid - 1
        left[idx] = lo
    return left


def fem1d_interpolate(node_x: np.ndarray, node_val: np.ndarray,
                      sample_x: np.ndarray) -> np.ndarray:
    """
    一维有限元分段线性插值，对应 398_fem1d_sample/fem1d_evaluate。

    对线性元 (element_order=2):
        phi(x) = val[l] * (x[r]-x) / (x[r]-x[l])
               + val[r] * (x-x[l]) / (x[r]-x[l])
    """
    node_x = np.asarray(node_x, dtype=float)
    node_val = np.asarray(node_val, dtype=float)
    sample_x = np.asarray(sample_x, dtype=float)
    left = bracket4(node_x, sample_x)
    sample_val = np.zeros(sample_x.shape[0])
    for i, l in enumerate(left):
        r = min(l + 1, node_x.shape[0] - 1)
        h = node_x[r] - node_x[l]
        if abs(h) < 1e-14:
            sample_val[i] = node_val[l]
        else:
            w_l = (node_x[r] - sample_x[i]) / h
            w_r = (sample_x[i] - node_x[l]) / h
            sample_val[i] = node_val[l] * w_l + node_val[r] * w_r
    return sample_val


def tetrahedron_volume(t: np.ndarray) -> float:
    """
    计算四面体体积，对应 418_fem3d_project/basis_mn_tet4 中的体积公式。

    顶点 t[:,0], t[:,1], t[:,2], t[:,3] (3x4 矩阵)
    Volume = |det([x1-x0, x2-x0, x3-x0])| / 6
    """
    t = np.asarray(t, dtype=float)
    if t.shape != (3, 4):
        raise ValueError("t必须是3x4矩阵")
    v1 = t[:, 1] - t[:, 0]
    v2 = t[:, 2] - t[:, 0]
    v3 = t[:, 3] - t[:, 0]
    vol = abs(np.linalg.det(np.vstack([v1, v2, v3]))) / 6.0
    return vol


def basis_mn_tet4(t: np.ndarray, n: int, p: np.ndarray) -> np.ndarray:
    """
    四面体线性基础函数求值，对应 418_fem3d_project/basis_mn_tet4。

    对于四面体顶点 t[:,0:4]，在点 p[:,0:n] 处求4个线性基础函数值。
    基础函数满足: phi_i(v_j) = delta_{ij} 且 sum(phi_i) = 1。

    参数:
        t: 3x4 顶点坐标
        n: 求值点数
        p: 3xn 求值点坐标
    返回:
        phi: 4xn 基础函数值
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if p.ndim == 1 and t.shape[0] == 3:
        p = p.reshape(3, 1)
        n = 1
    phi = np.zeros((4, n))

    # 体积 = det([[x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4],[1,1,1,1]])
    M = np.vstack([t, np.ones(4)])
    volume = np.linalg.det(M)
    if abs(volume) < 1e-14:
        raise ValueError("四面体体积为零")

    for k in range(n):
        pk = p[:, k] if n > 1 else p[:, 0]
        for i in range(4):
            Mi = M.copy()
            Mi[:3, i] = pk
            phi[i, k] = np.linalg.det(Mi) / volume
    return phi


def project_sample_to_fem3d(sample_nodes: np.ndarray, sample_vals: np.ndarray,
                            fem_nodes: np.ndarray, fem_elements: np.ndarray) -> np.ndarray:
    """
    将采样数据投影到3D FEM网格上，对应 418_fem3d_project/fem3d_transfer 思想。

    对每个FEM节点k，使用其邻接单元的采样点基础函数加权平均:
        val[k] = sum_{T\ni k} sum_{p in T} w_p * phi_k(p) * sample(p)
              / sum_{T\ni k} sum_{p in T} w_p * phi_k(p)
    """
    n_fem = fem_nodes.shape[0]
    fem_vals = np.zeros(n_fem)
    weights = np.zeros(n_fem)

    for elem in fem_elements:
        t = fem_nodes[elem, :].T  # 3x4
        try:
            vol = tetrahedron_volume(t)
        except ValueError:
            continue
        # 使用该单元的形心作为代表点
        centroid = np.mean(t, axis=1)
        # 寻找最近的采样点
        dists = np.linalg.norm(sample_nodes - centroid, axis=1)
        nearest = int(np.argmin(dists))
        val = sample_vals[nearest]

        # 基础函数在形心处均为1/4
        for i, node_idx in enumerate(elem):
            w = vol * 0.25
            fem_vals[node_idx] += w * val
            weights[node_idx] += w

    # 避免除零
    mask = weights > 1e-14
    fem_vals[mask] /= weights[mask]
    fem_vals[~mask] = 0.0
    return fem_vals


class FEMExpectationSampler:
    """
    有限元期望值采样器。
    将量子测量结果的概率分布离散化为有限元网格上的函数，
    通过有限元插值减少测量方差。
    """
    def __init__(self, n_qubits: int, n_grid_1d: int = 32):
        self.n_qubits = n_qubits
        self.n_grid_1d = n_grid_1d
        self.dim = 2 ** n_qubits
        # 一维参数网格（用于单参数期望值插值）
        self.theta_grid = np.linspace(-np.pi, np.pi, n_grid_1d)
        self.cache: dict = {}

    def estimate_energy_1d(self, theta_vals: np.ndarray, energy_vals: np.ndarray,
                           target_theta: float) -> float:
        """
        使用1D FEM插值估计目标参数处的能量。
        """
        return float(fem1d_interpolate(theta_vals, energy_vals,
                                       np.array([target_theta]))[0])

    def build_probability_density(self, bitstrings: np.ndarray,
                                   probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从测量结果构建离散概率密度函数（ histogram -> FEM nodes ）。
        """
        # 将比特串映射到 [0,1] 区间
        x_vals = bitstrings / (2 ** self.n_qubits - 1 + 1e-14)
        # 构建FEM节点和值
        nodes = np.linspace(0, 1, self.n_grid_1d)
        vals = np.zeros(self.n_grid_1d)
        counts = np.zeros(self.n_grid_1d)
        for x, p in zip(x_vals, probabilities):
            idx = int(np.clip(x * (self.n_grid_1d - 1), 0, self.n_grid_1d - 2))
            vals[idx] += p
            counts[idx] += 1
        mask = counts > 0
        vals[mask] /= counts[mask]
        return nodes, vals

    def integrate_expectation(self, nodes: np.ndarray, vals: np.ndarray,
                               observable_func: callable) -> float:
        """
        使用梯形法则（FEM积分的最低阶形式）计算期望值:
            <O> = integral O(x) * rho(x) dx
        """
        dx = np.diff(nodes)
        integrand = observable_func(nodes) * vals
        # 梯形积分
        result = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dx)
        return float(result)
