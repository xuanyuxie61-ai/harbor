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


class TetMesh:
    """
    四面体网格类，用于DNA-蛋白复合体的三维几何离散化。
    """

    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        """
        Parameters
        ----------
        nodes : ndarray, shape (N, 3)
            节点坐标。
        elements : ndarray, shape (M, 4)
            四面体单元节点索引（0-based）。
        """
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]

    def integrate_nodal_values(self, nodal_values: np.ndarray) -> tuple:
        """
        对节点值在四面体网格上进行体积积分。

        Returns
        -------
        integral : float
        total_volume : float
        """
        from tet_mesh_core import integrate_over_tet_mesh
        integral, total_volume = integrate_over_tet_mesh(
            self.nodes, self.elements, nodal_values
        )
        return integral, total_volume

    def compute_surface_area(self) -> float:
        """
        计算四面体网格的边界表面积。

        通过统计所有只被一个四面体使用的三角形面。
        """
        face_count = {}
        for e in range(self.n_elements):
            en = self.elements[e, :4]
            faces = [
                tuple(sorted([en[0], en[1], en[2]])),
                tuple(sorted([en[0], en[1], en[3]])),
                tuple(sorted([en[0], en[2], en[3]])),
                tuple(sorted([en[1], en[2], en[3]])),
            ]
            for f in faces:
                face_count[f] = face_count.get(f, 0) + 1

        surface_area = 0.0
        for f, count in face_count.items():
            if count == 1:
                p1, p2, p3 = self.nodes[f[0]], self.nodes[f[1]], self.nodes[f[2]]
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p1 - p3)
                s = 0.5 * (a + b + c)
                area = np.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
                surface_area += area
        return surface_area

    def to_xml_string(self) -> str:
        """
        将网格导出为简单的XML字符串。
        """
        lines = []
        lines.append('<?xml version="1.0"?>')
        lines.append('<TetMesh>')
        lines.append(f'  <Nodes count="{self.n_nodes}">')
        for i in range(self.n_nodes):
            x, y, z = self.nodes[i]
            lines.append(f'    <Node id="{i}" x="{x:.8e}" y="{y:.8e}" z="{z:.8e}"/>')
        lines.append('  </Nodes>')
        lines.append(f'  <Elements count="{self.n_elements}">')
        for i in range(self.n_elements):
            n1, n2, n3, n4 = self.elements[i]
            lines.append(f'    <Tet id="{i}" n0="{n1}" n1="{n2}" n2="{n3}" n3="{n4}"/>')
        lines.append('  </Elements>')
        lines.append('</TetMesh>')
        return '\n'.join(lines)


def generate_nucleosome_tet_mesh(
    n_rings: int = 4,
    n_theta: int = 8,
    n_z: int = 4,
    major_radius: float = 5.5,
    minor_radius: float = 3.3,
    pitch: float = 2.7,
) -> TetMesh:
    """
    生成简化核小体阵列的四面体网格。

    采用螺旋环面参数化，将规则盒子网格变形为环面螺旋形状。

    Parameters
    ----------
    n_rings : int
        环面径向分段数
    n_theta : int
        环面角度分段数
    n_z : int
        轴向（螺旋方向）分段数
    major_radius : float
        环面大半径 (nm)
    minor_radius : float
        环面小半径 (nm)
    pitch : float
        每圈的轴向螺距 (nm)

    Returns
    -------
    mesh : TetMesh
    """
    from tet_mesh_core import generate_tet_mesh_box

    # 先生成一个规则盒子网格
    nodes_box, elements_box = generate_tet_mesh_box(
        nx=n_rings + 1,
        ny=n_theta + 1,
        nz=n_z + 1,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        zlim=(0.0, 2.0 * np.pi * n_rings),
    )

    # 将盒子变形为环面螺旋
    nodes = nodes_box.copy()
    for i in range(nodes.shape[0]):
        u = nodes_box[i, 0]  # [-1, 1]
        v = nodes_box[i, 1]  # [-1, 1]
        w = nodes_box[i, 2]  # [0, 2π * n_rings]

        # 将 u, v 映射到环面截面
        theta_ring = np.arctan2(v, u) if (abs(u) > 1e-12 or abs(v) > 1e-12) else 0.0
        r_frac = np.sqrt(u * u + v * v)
        r_local = minor_radius * r_frac

        # 螺旋角
        phi = w / (2.0 * np.pi * n_rings) * 2.0 * np.pi * n_rings
        # 实际上是沿螺旋线的角度
        angle = w

        R_eff = major_radius + r_local * np.cos(theta_ring)
        nodes[i, 0] = R_eff * np.cos(angle)
        nodes[i, 1] = R_eff * np.sin(angle)
        nodes[i, 2] = pitch * angle / (2.0 * np.pi) + r_local * np.sin(theta_ring)

    return TetMesh(nodes, elements_box)


def compute_dsb_repair_compartment_volume(
    mesh: TetMesh,
    gamma_density: np.ndarray,
    threshold: float = 0.15,
) -> tuple:
    """
    计算 γH2AX 密度超过阈值的修复腔室体积。

    对每个四面体，若其四个顶点平均密度超过阈值，则将其体积计入。

    Parameters
    ----------
    mesh : TetMesh
    gamma_density : ndarray, shape (n_nodes,)
        每个节点上的 γH2AX 相对密度。
    threshold : float
        阈值。

    Returns
    -------
    compartment_volume : float
    total_signal : float
    """
    gamma_density = np.asarray(gamma_density, dtype=np.float64)
    compartment_volume = 0.0
    total_signal = 0.0

    for e in range(mesh.n_elements):
        en = mesh.elements[e, :4]
        p = mesh.nodes[en]
        M = np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))
        vol = abs(np.linalg.det(M)) / 6.0
        avg_density = np.mean(gamma_density[en])
        total_signal += vol * avg_density
        if avg_density > threshold:
            compartment_volume += vol

    return compartment_volume, total_signal
