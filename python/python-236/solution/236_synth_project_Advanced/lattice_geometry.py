"""
lattice_geometry.py — 四维周期性超立方体格点的几何与拓扑结构
================================================================
融合种子项目:
  [185_circles]    : 周期性边界下的圆邻域(超球)邻居枚举
  [305_dist_plot]  : 有符号距离函数 → 格点间测地距离
  [1422_xyl_display]: 格点坐标与链接索引的序列化 I/O

物理背景:
  在格点 QCD 中, 欧几里得时空被离散化为四维超立方体格点
  Lambda = (L_s)^3 x L_t,  其中 L_s 为空间尺寸, L_t 为时间尺寸.
  每个格点 x = (x_0, x_1, x_2, x_3) 与四条链接 (link) 相连,
  链接沿 mu 方向连接到 x + hat{mu}.
  周期性边界条件: x_mu ~ x_mu + L_mu.

核心公式:
  格点总数:  V = prod_{mu=0}^{3} L_mu
  线性索引:  idx(x) = x_0 + L_0*(x_1 + L_1*(x_2 + L_2*x_3))
  测地距离:  d_geo(x,y) = sqrt(sum_mu min(|x_mu-y_mu|, L_mu-|x_mu-y_mu|)^2)
  超球邻居:  N_r(x) = {y : d_geo(x,y) <= r}
"""

import numpy as np
from typing import Tuple, List, Optional
import json
import os


class LatticeGeometry:
    """四维周期性超立方体格点几何.

    参数
    ----
    Ls : int
        空间尺寸 (各方向相同), 典型值 4, 6, 8.
    Lt : int
        时间尺寸, 典型值 8, 12, 16.
    ndim : int
        时空维度 (默认 4).
    """

    def __init__(self, Ls: int = 4, Lt: int = 8, ndim: int = 4):
        if Ls < 2:
            raise ValueError(f"空间尺寸 Ls={Ls} 至少为 2 (需要 PBC)")
        if Lt < 2:
            raise ValueError(f"时间尺寸 Lt={Lt} 至少为 2")
        if ndim < 2:
            raise ValueError(f"维度 ndim={ndim} 至少为 2")

        self.Ls = Ls
        self.Lt = Lt
        self.ndim = ndim
        # 各方向尺寸: 前 (ndim-1) 个为 Ls, 最后一个为 Lt
        self.shape = tuple([Ls] * (ndim - 1) + [Lt])
        self.volume = int(np.prod(self.shape))
        self._ndim = ndim

        # 预计算线性索引映射
        self._strides = np.zeros(ndim, dtype=np.int64)
        self._strides[0] = 1
        for mu in range(1, ndim):
            self._strides[mu] = self._strides[mu - 1] * self.shape[mu - 1]

    # ------------------------------------------------------------------
    # 坐标 <-> 线性索引 转换
    # ------------------------------------------------------------------
    def coord_to_index(self, coord: np.ndarray) -> int:
        """将格点坐标转换为线性索引.

        idx(x) = sum_{mu} x_mu * strides[mu]
        """
        coord = np.asarray(coord, dtype=np.int64)
        if coord.shape != (self.ndim,):
            raise ValueError(f"坐标维度 {coord.shape} 不匹配 {self.ndim}")
        # 周期性约化
        coord = coord % np.array(self.shape, dtype=np.int64)
        return int(np.dot(coord, self._strides))

    def index_to_coord(self, idx: int) -> np.ndarray:
        """将线性索引转换为格点坐标."""
        if idx < 0 or idx >= self.volume:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.volume})")
        coord = np.zeros(self.ndim, dtype=np.int64)
        rem = idx
        for mu in range(self.ndim - 1, -1, -1):
            coord[mu] = rem // self._strides[mu]
            rem = rem % self._strides[mu]
        return coord

    # ------------------------------------------------------------------
    # 周期性距离 (源自 [305_dist_plot] 的距离场思想)
    # ------------------------------------------------------------------
    def pbc_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算周期性边界条件下的测地距离.

        d_geo(x,y) = sqrt(sum_mu min(|dx_mu|, L_mu - |dx_mu|)^2)

        这源自 DistMesh 中有符号距离函数在周期性域上的推广.
        """
        dx = np.abs(np.asarray(x) - np.asarray(y))
        shape = np.array(self.shape, dtype=np.float64)
        dx = np.minimum(dx, shape - dx)
        return float(np.sqrt(np.sum(dx ** 2)))

    def pbc_distance_sq(self, x: np.ndarray, y: np.ndarray) -> float:
        """距离的平方 (避免开方, 用于邻居判断)."""
        dx = np.abs(np.asarray(x) - np.asarray(y))
        shape = np.array(self.shape, dtype=np.float64)
        dx = np.minimum(dx, shape - dx)
        return float(np.sum(dx ** 2))

    # ------------------------------------------------------------------
    # 超球邻居枚举 (源自 [185_circles] 的圆形邻域)
    # ------------------------------------------------------------------
    def sphere_neighbors(self, center: np.ndarray, radius: float
                         ) -> List[Tuple[np.ndarray, float]]:
        """枚举半径为 r 的超球内所有格点邻居 (源自 [185_circles]).

        对格点 center, 返回所有满足 d_geo(center, y) <= r 的格点 y
        及其距离. 这类似于 circles.m 中以 (x,y) 为圆心 r 为半径
        画圆的概念, 但推广到四维周期性格点.

        参数
        ----
        center : array_like, shape (ndim,)
        radius : float

        返回
        ----
        neighbors : list of (coord, distance)
        """
        center = np.asarray(center, dtype=np.int64)
        r_sq = radius ** 2
        results = []
        # 枚举搜索范围
        ranges = [range(-int(radius) - 1, int(radius) + 2) for _ in range(self.ndim)]
        for offsets in np.array(np.meshgrid(*ranges)).T.reshape(-1, self.ndim):
            y = center + offsets
            d_sq = self.pbc_distance_sq(center, y)
            if d_sq <= r_sq + 1e-12:
                y_pbc = y % np.array(self.shape)
                results.append((y_pbc, float(np.sqrt(d_sq))))
        return results

    # ------------------------------------------------------------------
    # 正方向邻居 (链接)
    # ------------------------------------------------------------------
    def neighbor_index(self, idx: int, mu: int, direction: int = +1) -> int:
        """沿方向 mu 的最近邻格点索引 (direction=+1 为正, -1 为负).

        对应格点 QCD 中的链接 (link): U_mu(x) 连接 x 和 x+hat{mu}.
        """
        if mu < 0 or mu >= self.ndim:
            raise ValueError(f"方向 mu={mu} 超出 [0, {self.ndim})")
        coord = self.index_to_coord(idx)
        coord[mu] = (coord[mu] + direction) % self.shape[mu]
        return self.coord_to_index(coord)

    # ------------------------------------------------------------------
    # 格点坐标数组 (所有格点)
    # ------------------------------------------------------------------
    def all_coords(self) -> np.ndarray:
        """返回所有格点坐标, shape = (volume, ndim)."""
        grids = [np.arange(s) for s in self.shape]
        mesh = np.meshgrid(*grids, indexing='ij')
        coords = np.stack([m.ravel() for m in mesh], axis=1)
        return coords

    # ------------------------------------------------------------------
    # 动量空间 (离散 Fourier 变换)
    # ------------------------------------------------------------------
    def momentum_vector(self, n: np.ndarray) -> np.ndarray:
        """计算格点动量 (周期性边界 → 离散动量).

        p_mu = (2*pi / L_mu) * n_mu,   n_mu in [0, L_mu)

        在格点 QCD 中, 动量空间是离散的 Brillouin 区:
        p_mu in (-pi/a, pi/a], 取 a=1 晶格单位.
        """
        n = np.asarray(n, dtype=np.float64)
        shape = np.array(self.shape, dtype=np.float64)
        return 2.0 * np.pi * n / shape

    def momentum_squared(self, n: np.ndarray) -> float:
        """动量平方 p^2 = sum_mu p_mu^2."""
        p = self.momentum_vector(n)
        return float(np.sum(p ** 2))

    # ------------------------------------------------------------------
    # I/O (源自 [1422_xyl_display] 的坐标/链接文件读写)
    # ------------------------------------------------------------------
    def save_connectivity(self, filepath: str):
        """保存格点连通性到 JSON 文件 (源自 [1422_xyl_display]).

        格式类似 .xy/.xyl 文件: 存储格点坐标和链接索引对.
        """
        data = {
            'shape': list(self.shape),
            'volume': self.volume,
            'ndim': self.ndim,
            'links': []
        }
        for idx in range(min(self.volume, 256)):  # 限制大小用于可重现实验
            coord = self.index_to_coord(idx)
            links = []
            for mu in range(self.ndim):
                fwd = self.neighbor_index(idx, mu, +1)
                bwd = self.neighbor_index(idx, mu, -1)
                links.append({'mu': mu, 'fwd': fwd, 'bwd': bwd})
            data['links'].append({
                'idx': idx,
                'coord': coord.tolist(),
                'links': links
            })
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=1)

    def __repr__(self) -> str:
        return (f"LatticeGeometry(Ls={self.Ls}, Lt={self.Lt}, "
                f"ndim={self.ndim}, V={self.volume})")


def compute_distance_matrix(geo: LatticeGeometry,
                            ref_point: Optional[np.ndarray] = None
                            ) -> np.ndarray:
    """计算从参考点到所有格点的距离矩阵.

    源自 [305_dist_plot] 的有符号距离场概念: 在格点上构建
    以 ref_point 为源的距离函数 d(x), 用于关联函数的空间衰减分析.

    返回
    ----
    dist : ndarray, shape (volume,)
        每个格点到参考点的距离.
    """
    if ref_point is None:
        ref_point = np.zeros(geo.ndim, dtype=np.int64)
    ref_point = np.asarray(ref_point, dtype=np.int64)
    coords = geo.all_coords()
    dist = np.zeros(geo.volume)
    for i in range(geo.volume):
        dist[i] = geo.pbc_distance(ref_point, coords[i])
    return dist
