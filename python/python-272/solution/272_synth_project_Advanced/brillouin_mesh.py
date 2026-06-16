"""
Brillouin 区网格生成与三角剖分模块
实现 Fukui-Hatsugai-Suzuki 方法所需的网格结构

物理背景：
Fukui-Hatsugai-Suzuki (FHS) 方法 [J. Phys. Soc. Jpn. 74, 1674 (2005)]
将布里渊区离散化为三角网格，通过计算每个小三角上的 U(1) 联络
来精确计算 Chern 数，保证整数化。
"""

import numpy as np
from typing import Tuple, List


class BrillouinMesh:
    """
    三维 Brillouin 区网格生成器

    支持：
    1. 均匀矩形网格 (用于有限差分)
    2. 三角剖分网格 (用于 FHS 方法)
    3. 自适应细化 (在 Weyl 点附近)
    """

    def __init__(self, lattice_vectors: np.ndarray = None):
        """
        初始化 Brillouin 区网格

        Parameters:
        -----------
        lattice_vectors : np.ndarray
            实空间晶格矢量 (3×3 矩阵，行矢量为 a1, a2, a3)
        """
        if lattice_vectors is None:
            # 简单立方晶格
            self.a1 = np.array([1.0, 0.0, 0.0])
            self.a2 = np.array([0.0, 1.0, 0.0])
            self.a3 = np.array([0.0, 0.0, 1.0])
        else:
            self.a1 = lattice_vectors[0]
            self.a2 = lattice_vectors[1]
            self.a3 = lattice_vectors[2]

        # 计算倒格矢 (b_i · a_j = 2π δ_ij)
        self.volume = np.abs(np.dot(self.a1, np.cross(self.a2, self.a3)))
        self.b1 = 2 * np.pi * np.cross(self.a2, self.a3) / self.volume
        self.b2 = 2 * np.pi * np.cross(self.a3, self.a1) / self.volume
        self.b3 = 2 * np.pi * np.cross(self.a1, self.a2) / self.volume

        self.reciprocal_vectors = np.array([self.b1, self.b2, self.b3])

    def generate_uniform_mesh(self, n1: int, n2: int, n3: int,
                              fractional: bool = True) -> Tuple[np.ndarray, dict]:
        """
        生成均匀 k 空间网格

        k = (i/n1)·b1 + (j/n2)·b2 + (k/n3)·b3

        Parameters:
        -----------
        n1, n2, n3 : int
            三个倒格子方向的分割数
        fractional : bool
            是否返回分数坐标

        Returns:
        --------
        Tuple[np.ndarray, dict]
            k_points: (N, 3) 网格点坐标
            mesh_info: 网格信息字典
        """
        k_points = []
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    if fractional:
                        k_vec = np.array([i/n1, j/n2, k/n3])
                    else:
                        k_vec = (i/n1) * self.b1 + (j/n2) * self.b2 + (k/n3) * self.b3
                    k_points.append(k_vec)

        k_points = np.array(k_points)

        mesh_info = {
            'n1': n1, 'n2': n2, 'n3': n3,
            'total_points': n1 * n2 * n3,
            'fractional': fractional,
            'differential_volume': (2*np.pi)**3 / (n1 * n2 * n3)
        }

        return k_points, mesh_info

    def generate_triangular_mesh_2d(self, plane: str = 'kz_const',
                                     kz_value: float = 0.0,
                                     n1: int = 20, n2: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成二维三角剖分网格 (用于计算切片 Chern 数)

        Parameters:
        -----------
        plane : str
            切片平面 ('kx_const', 'ky_const', 'kz_const')
        kz_value : float
            固定坐标值 (分数坐标)
        n1, n2 : int
            网格密度

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            vertices: 网格顶点
            triangles: (M, 3) 三角连接关系
        """
        vertices = []
        for i in range(n1 + 1):
            for j in range(n2 + 1):
                if plane == 'kz_const':
                    v = np.array([i/n1, j/n2, kz_value])
                elif plane == 'kx_const':
                    v = np.array([kz_value, i/n1, j/n2])
                elif plane == 'ky_const':
                    v = np.array([i/n1, kz_value, j/n2])
                vertices.append(v)

        vertices = np.array(vertices)

        # 构建三角连接
        triangles = []
        for i in range(n1):
            for j in range(n2):
                # 每个矩形单元分为两个三角形
                idx00 = i * (n2 + 1) + j
                idx10 = (i + 1) * (n2 + 1) + j
                idx01 = i * (n2 + 1) + (j + 1)
                idx11 = (i + 1) * (n2 + 1) + (j + 1)

                triangles.append([idx00, idx10, idx01])
                triangles.append([idx10, idx11, idx01])

        triangles = np.array(triangles)
        return vertices, triangles

    def adaptive_refine_around_weyl(self, weyl_position: np.ndarray,
                                    radius: float = 0.5,
                                    base_n: int = 10,
                                    refine_factor: int = 4) -> Tuple[np.ndarray, dict]:
        """
        在 Weyl 点附近自适应细化网格

        细化准则：|k - k_W| < R 的区域使用 4× 密度

        Parameters:
        -----------
        weyl_position : np.ndarray
            Weyl 点位置 (分数坐标)
        radius : float
            细化半径
        base_n : int
            基础网格密度
        refine_factor : int
            细化倍数

        Returns:
        --------
        Tuple[np.ndarray, dict]
            k_points: 自适应网格点
            weights: 每个点的积分权重
        """
        # 粗网格
        k_coarse, _ = self.generate_uniform_mesh(base_n, base_n, base_n)

        # 细网格区域
        n_fine = base_n * refine_factor
        k_fine, _ = self.generate_uniform_mesh(n_fine, n_fine, n_fine)

        # 筛选细化区域内的点
        dist = np.linalg.norm(k_fine - weyl_position, axis=1)
        mask_inner = dist < radius

        # 粗网格中不在细化区域的点
        dist_coarse = np.linalg.norm(k_coarse - weyl_position, axis=1)
        mask_coarse = dist_coarse >= radius

        # 合并
        k_points = np.vstack([k_coarse[mask_coarse], k_fine[mask_inner]])

        # 计算权重 (粗网格权重 1，细网格权重 1/refine_factor^3)
        weights = np.ones(len(k_points))
        weights[len(k_coarse[mask_coarse]):] = 1.0 / (refine_factor**3)

        mesh_info = {
            'n_coarse': np.sum(mask_coarse),
            'n_fine': np.sum(mask_inner),
            'total_points': len(k_points),
            'weyl_position': weyl_position,
            'refine_radius': radius
        }

        return k_points, mesh_info

    def get_link_path(self, k_start: np.ndarray, k_end: np.ndarray,
                     n_steps: int = 10) -> np.ndarray:
        """
        获取两点间的高对称路径 (用于能带计算)

        Parameters:
        -----------
        k_start, k_end : np.ndarray
            起止 k 点
        n_steps : int
            路径步数

        Returns:
        --------
        np.ndarray
            路径上的 k 点序列
        """
        t = np.linspace(0, 1, n_steps)
        path = np.outer(1 - t, k_start) + np.outer(t, k_end)
        return path

    def high_symmetry_path(self, special_points: dict = None) -> Tuple[np.ndarray, List[str]]:
        """
        生成高对称路径 (用于能带结构计算)

        标准立方 Brillouin 区高对称点:
        Γ = (0,0,0), X = (π,0,0), M = (π,π,0), R = (π,π,π)

        Parameters:
        -----------
        special_points : dict
            自定义高对称点

        Returns:
        --------
        Tuple[np.ndarray, List[str]]
            k_path: 路径上的 k 点
            labels: 高对称点标签
        """
        if special_points is None:
            special_points = {
                'Gamma': np.array([0.0, 0.0, 0.0]),
                'X': np.array([0.5, 0.0, 0.0]),
                'M': np.array([0.5, 0.5, 0.0]),
                'R': np.array([0.5, 0.5, 0.5]),
                'W': np.array([0.5, 0.25, 0.75])
            }

        # 标准路径 Γ-X-M-Γ-R-X|R-M
        path_order = ['Gamma', 'X', 'M', 'Gamma', 'R', 'X', 'R', 'M']
        n_steps = 50

        k_path = []
        labels = []

        for i in range(len(path_order) - 1):
            k_start = special_points[path_order[i]]
            k_end = special_points[path_order[i + 1]]
            segment = self.get_link_path(k_start, k_end, n_steps)
            k_path.append(segment)
            if i == 0:
                labels.append(path_order[i])
            labels.append(path_order[i + 1])

        k_path = np.vstack(k_path)
        return k_path, labels

    def compute_mesh_quality(self, k_points: np.ndarray, triangles: np.ndarray = None) -> dict:
        """
        计算网格质量指标

        Parameters:
        -----------
        k_points : np.ndarray
            网格点
        triangles : np.ndarray
            三角连接 (可选)

        Returns:
        --------
        dict
            网格质量指标
        """
        # 最小间距
        from scipy.spatial.distance import pdist
        if len(k_points) > 1:
            distances = pdist(k_points)
            min_dist = np.min(distances)
            mean_dist = np.mean(distances)
        else:
            min_dist = 0.0
            mean_dist = 0.0

        quality = {
            'n_points': len(k_points),
            'min_distance': min_dist,
            'mean_distance': mean_dist,
            'bounding_box': np.ptp(k_points, axis=0)
        }

        if triangles is not None:
            # 计算三角形质量 (面积)
            areas = []
            for tri in triangles:
                v0 = k_points[tri[0]]
                v1 = k_points[tri[1]]
                v2 = k_points[tri[2]]
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                areas.append(area)
            quality['n_triangles'] = len(triangles)
            quality['min_area'] = np.min(areas) if areas else 0.0
            quality['mean_area'] = np.mean(areas) if areas else 0.0

        return quality
