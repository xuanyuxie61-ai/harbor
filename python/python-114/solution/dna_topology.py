"""
dna_topology.py
DNA粗粒化拓扑与几何建模模块

融合原项目:
  - 1052_sammon_data: 螺旋/圆/线性数据生成 → DNA双螺旋骨架坐标生成
  - 578_image_double: 图像分辨率加倍 → 粗粒化映射密度网格细化

科学背景:
  在DNA损伤修复的分子动力学模拟中，单链DNA(ssDNA)被粗粒化为珠子链模型。
  每个珠子代表3个核苷酸(约1nm)。DNA骨架遵循螺旋几何:
    r(t) = (R cos(ωt), R sin(ωt), p·t)
  其中 R 为螺旋半径(~1nm)，p 为螺距(~0.34nm/碱基对)，ω = 2π/T 为角频率。
"""

import numpy as np


class DnaCoarseGrainTopology:
    """
    粗粒化DNA拓扑生成器
    """

    def __init__(self, num_bases: int = 180, bases_per_bead: int = 3,
                 helix_radius_nm: float = 1.0, rise_per_base_nm: float = 0.34,
                 bases_per_turn: float = 10.5):
        """
        参数:
            num_bases: DNA单链总碱基数 (典型ssDNA约180nt)
            bases_per_bead: 每个粗粒化珠子代表的碱基数
            helix_radius_nm: 螺旋半径 (nm)
            rise_per_base_nm: 每碱基轴向上升距离 (nm)
            bases_per_turn: 每圈碱基数
        """
        if num_bases <= 0:
            raise ValueError("num_bases must be positive")
        if bases_per_bead <= 0:
            raise ValueError("bases_per_bead must be positive")
        if helix_radius_nm <= 0 or rise_per_base_nm <= 0 or bases_per_turn <= 0:
            raise ValueError("Geometric parameters must be positive")

        self.num_bases = num_bases
        self.bases_per_bead = bases_per_bead
        self.helix_radius = helix_radius_nm
        self.rise_per_base = rise_per_base_nm
        self.bases_per_turn = bases_per_turn
        self.num_beads = int(np.ceil(num_bases / bases_per_bead))

    def generate_helix_coordinates(self) -> np.ndarray:
        """
        生成DNA单链螺旋骨架的三维坐标

        螺旋参数方程:
            x_i = R * cos(2π * i / N_turn)
            y_i = R * sin(2π * i / N_turn)
            z_i = i * h
        其中 i 为珠子索引，h = bases_per_bead * rise_per_base 为每珠轴向位移,
        N_turn = bases_per_turn / bases_per_bead 为每圈珠子数。

        Returns:
            coords: shape (num_beads, 3) 的坐标数组 (单位: nm)
        """
        n = self.num_beads
        t = np.arange(n, dtype=float)
        angular_freq = 2.0 * np.pi / (self.bases_per_turn / self.bases_per_bead)
        h = self.bases_per_bead * self.rise_per_base

        x = self.helix_radius * np.cos(angular_freq * t)
        y = self.helix_radius * np.sin(angular_freq * t)
        z = h * t

        coords = np.column_stack((x, y, z))
        return coords

    def generate_backbone_tangents(self) -> np.ndarray:
        """
        计算每个珠子处的骨架切向量（归一化）

        对参数方程 r(t) 求导:
            dr/dt = (-Rω sin(ωt), Rω cos(ωt), h)
            |dr/dt| = sqrt(R²ω² + h²)

        Returns:
            tangents: shape (num_beads, 3)
        """
        coords = self.generate_helix_coordinates()
        n = self.num_beads
        tangents = np.zeros_like(coords)

        # 内部点: 中心差分
        for i in range(1, n - 1):
            tangents[i] = coords[i + 1] - coords[i - 1]
            norm = np.linalg.norm(tangents[i])
            if norm > 1e-12:
                tangents[i] /= norm

        # 边界: 前向/后向差分
        if n > 1:
            tangents[0] = coords[1] - coords[0]
            tangents[-1] = coords[-1] - coords[-2]
            for idx in (0, -1):
                norm = np.linalg.norm(tangents[idx])
                if norm > 1e-12:
                    tangents[idx] /= norm
        else:
            tangents[0] = np.array([0.0, 0.0, 1.0])

        return tangents

    def compute_persistence_length_correction(self, persistence_length_nm: float = 3.0) -> np.ndarray:
        """
        基于 persistence length 的谐波约束刚度矩阵对角元

        对于 worm-like chain 模型，弯曲能:
            E_bend = (k_B T * l_p / 2) * ∫ (d²r/ds²)² ds
        离散化后每个键的等效弹簧常数:
            k_bend = k_B T * l_p / (Δs)³
        其中 Δs = bases_per_bead * 0.34 nm 为键长

        Returns:
            stiffness: shape (num_beads,) 每个珠子的弯曲刚度 (kJ/mol/nm²)
        """
        kB_T = 2.479  # kJ/mol at 298K
        delta_s = self.bases_per_bead * self.rise_per_base  # nm
        if delta_s <= 0:
            raise ValueError("bond length must be positive")
        k_bend = kB_T * persistence_length_nm / (delta_s ** 3)
        return np.full(self.num_beads, k_bend)

    def double_resolution_grid(self, grid_1d: np.ndarray) -> np.ndarray:
        """
        基于 image_double 思想的一维网格分辨率加倍
        用于将粗粒化珠子密度映射到细网格

        参数:
            grid_1d: 原始一维数组
        Returns:
            分辨率加倍后的数组 (每个元素复制为相邻两个相同值)
        """
        if grid_1d.size == 0:
            return grid_1d.copy()
        m = grid_1d.shape[0]
        if grid_1d.ndim == 1:
            out = np.repeat(grid_1d, 2)
        else:
            # 对多维数组沿第一维加倍
            out = np.zeros((2 * m,) + grid_1d.shape[1:], dtype=grid_1d.dtype)
            for i in range(m):
                out[2 * i] = grid_1d[i]
                out[2 * i + 1] = grid_1d[i]
        return out

    def generate_ssdna_with_bubble(self, bubble_start: int = 60, bubble_length: int = 30) -> dict:
        """
        生成带有单链泡(ssDNA bubble)的DNA结构，模拟DSB暴露区域

        Returns:
            dict 包含 'coords', 'is_bubble', 'binding_sites'
        """
        coords = self.generate_helix_coordinates()
        n = self.num_beads
        is_bubble = np.zeros(n, dtype=bool)

        # 标记bubble区域
        b_start = max(0, int(bubble_start / self.bases_per_bead))
        b_end = min(n, int((bubble_start + bubble_length) / self.bases_per_bead))
        is_bubble[b_start:b_end] = True

        # 结合位点: 非bubble区域有基础结合能力，bubble区域有增强结合能力
        binding_sites = np.ones(n, dtype=float) * 0.1
        binding_sites[is_bubble] = 1.0

        return {
            'coords': coords,
            'is_bubble': is_bubble,
            'binding_sites': binding_sites,
            'num_beads': n,
            'bubble_indices': np.arange(b_start, b_end)
        }


def generate_sammon_helix(n_points: int, radius: float = 1.0, pitch: float = 3.4) -> np.ndarray:
    """
    基于 sammon_data 中 data_helix 的螺旋数据生成思想，
    生成用于降维分析测试的高维DNA构象数据

    参数方程:
        z = t / sqrt(2)
        x = R * cos(z)
        y = R * sin(z)
    """
    z = np.arange(n_points, dtype=float) / np.sqrt(2.0)
    x = radius * np.cos(z)
    y = radius * np.sin(z)
    return np.column_stack((x, y, z * pitch / 3.4))


def compute_worm_like_chain_end_to_end(num_bases: int, persistence_length_nm: float = 3.0,
                                        base_rise_nm: float = 0.34) -> float:
    """
    Worm-like chain 模型的末端距均方根

    理论公式:
        <R²> = 2 * L_p * L * [1 - L_p/L * (1 - exp(-L/L_p))]
    其中 L = num_bases * base_rise_nm 为轮廓长度，L_p 为 persistence length

    Returns:
        RMS末端距 (nm)
    """
    L = num_bases * base_rise_nm
    if L <= 0:
        return 0.0
    if persistence_length_nm <= 0:
        return L
    ratio = L / persistence_length_nm
    rms = np.sqrt(2.0 * persistence_length_nm * L * (1.0 - (1.0 - np.exp(-ratio)) / ratio))
    return rms
