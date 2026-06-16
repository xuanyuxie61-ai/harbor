"""
detector_geometry.py — 柱状追踪探测器几何与材料层描述
=====================================================

本模块构建多层柱状追踪探测器的几何描述，融合以下种子项目算法:

    [067_ball_grid]     : 3D 球坐标网格生成 → 用于探测器灵敏体积的等距采样
    [1423_xyz_display]  : 3D 点云边界归一化 → 用于击中点坐标的范围归一化
    [1322_triangle_to_xml]: 三角网格序列化 → 用于探测器几何的 XML 格式导出
    [1330_triangulation]: Delaunay 三角化质量度量 → 用于评估探测器层覆盖质量
    [503_hand_mesh2d]   : 2D 非结构网格生成 → 用于探测器横截面的网格细化

核心物理模型:
    柱状追踪探测器由 N 个同轴圆柱层组成，每层定义:
        - 半径 r [mm]
        - 半长 L/2 [mm]
        - 材料厚度 t/X₀ (以辐射长度分数表示)
        - 空间分辨率 σ_r, σ_z [mm]
        - 材料成分

粒子穿过第 k 层时经历的物理效应:
    多次散射角 (Highland 公式):
        θ₀ = (13.6 MeV / βcp) · z · √(t/X₀) · [1 + 0.038·ln(t/X₀)]

    最概然能量损失 (Landau-Vavilov):
        Δ_p = ξ · [ln(2m_e c² β²γ² / I) + ln(ξ / E_max) + 0.2 - β²]
        其中 ξ = (K/2) · (Z/A) · ρ · t / β²
"""

import math
from typing import List, Tuple, Optional

# ============================================================
# 探测器层数据结构
# ============================================================
class DetectorLayer:
    """
    描述柱状追踪探测器的单个灵敏层

    Attributes
    ----------
    layer_id : int
        层编号 (从内向外递增)
    radius_mm : float
        层半径 [mm]
    half_length_mm : float
        层半长 [mm]
    thickness_ratio : float
        材料厚度 t/X₀ (辐射长度分数)
    sigma_r_mm : float
        径向空间分辨率 [mm]
    sigma_z_mm : float
        纵向空间分辨率 [mm]
    material : str
        材料名称
    radiation_length_cm : float
        辐射长度 X₀ [cm]
    density_g_cm3 : float
        密度 ρ [g/cm³]
    atomic_number : int
        原子序数 Z
    atomic_mass : float
        原子量 A [g/mol]
    ionization_eV : float
        平均激发能 I [eV]
    """

    def __init__(self, layer_id, radius_mm, half_length_mm,
                 thickness_ratio, sigma_r_mm, sigma_z_mm,
                 material='silicon'):
        self.layer_id = layer_id
        self.radius_mm = radius_mm
        self.half_length_mm = half_length_mm
        self.thickness_ratio = thickness_ratio
        self.sigma_r_mm = sigma_r_mm
        self.sigma_z_mm = sigma_z_mm
        self.material = material

        # 从材料查表获取参数
        import constants as const
        self.radiation_length_cm = const.RADIATION_LENGTH.get(material, 9.37)
        self.density_g_cm3 = const.MATERIAL_DENSITY.get(material, 2.329)
        self.atomic_number = const.ATOMIC_NUMBER.get(material, 14)
        self.atomic_mass = const.ATOMIC_MASS.get(material, 28.085)
        self.ionization_eV = const.IONIZATION_POTENTIAL.get(material, 173.0)

    def surface_area_mm2(self):
        """计算层表面积: A = 2πrL"""
        return 2.0 * math.pi * self.radius_mm * 2.0 * self.half_length_mm

    def thickness_cm(self):
        """将辐射长度分数转换为绝对厚度 [cm]"""
        return self.thickness_ratio * self.radiation_length_cm

    def __repr__(self):
        return (f"DetectorLayer(id={self.layer_id}, r={self.radius_mm:.1f}mm, "
                f"t/X0={self.thickness_ratio:.4f}, σ_r={self.sigma_r_mm*1e3:.1f}μm)")


class TrackingDetector:
    """
    多层柱状追踪探测器

    管理一组 DetectorLayer 并提供:
    - 按半径排序的层访问
    - 材料预算查询
    - 击中点生成
    - 几何序列化
    """

    def __init__(self, layers: List[DetectorLayer]):
        self.layers = sorted(layers, key=lambda l: l.radius_mm)
        self.n_layers = len(self.layers)
        self._build_index()

    def _build_index(self):
        """构建层索引映射"""
        self._radius_list = [l.radius_mm for l in self.layers]
        self._radiation_lengths = [l.thickness_ratio for l in self.layers]
        self._total_material_budget = sum(self._radiation_lengths)

    @property
    def inner_radius(self):
        """最内层半径 [mm]"""
        return self.layers[0].radius_mm

    @property
    def outer_radius(self):
        """最外层半径 [mm]"""
        return self.layers[-1].radius_mm

    @property
    def total_material_budget(self):
        """总材料预算 [辐射长度分数]"""
        return self._total_material_budget

    def get_layer(self, idx):
        """获取指定层"""
        return self.layers[idx]

    def find_enclosing_layers(self, radius_mm):
        """
        找到给定半径两侧的最近层

        Returns
        -------
        tuple : (inner_layer, outer_layer)
            如果半径超出范围，对应位置返回 None
        """
        inner = None
        outer = None
        for layer in self.layers:
            if layer.radius_mm <= radius_mm:
                inner = layer
            else:
                outer = layer
                break
        return inner, outer

    def interpolate_material(self, radius_mm):
        """
        在给定半径处插值材料预算密度

        使用线性插值估计层间材料:
            (t/X₀)(r) ≈ Σ_k (t_k/X₀_k) · δ(r - r_k)

        对于连续近似，将离散层视为均匀分布:
            ρ_mat(r) = (t/X₀)_total / (r_out - r_in)
        """
        if radius_mm < self.inner_radius or radius_mm > self.outer_radius:
            return 0.0
        dr = self.outer_radius - self.inner_radius
        if dr < 1e-10:
            return self._total_material_budget
        return self._total_material_budget / dr

    # ============================================================
    # [067_ball_grid] 3D 网格采样 → 探测器灵敏体积采样
    # ============================================================
    def sample_sensitive_volume(self, n_radial, n_phi, n_z):
        """
        基于 [067_ball_grid] 的八分对称思想，在柱状探测器灵敏体积内
        生成均匀的测试点网格

        原算法在球体内利用 8 个象限的镜像对称性；这里适配为柱坐标下的
        (r, φ, z) 网格，并仅生成有效区域(层间空隙)中的点

        Parameters
        ----------
        n_radial : int
            径向分格数
        n_phi : int
            方位角分格数
        n_z : int
            纵向分格数

        Returns
        -------
        list of tuple
            [(r, phi, z), ...] 柱坐标点列表
        """
        r_min = self.inner_radius * 0.9
        r_max = self.outer_radius * 1.1
        z_max = self.layers[0].half_length_mm

        dr = (r_max - r_min) / max(n_radial, 1)
        dphi = 2.0 * math.pi / max(n_phi, 1)
        dz = 2.0 * z_max / max(n_z, 1)

        grid_points = []

        # 类似 [067] 的主循环: 在 (r, phi, z) 空间遍历
        for ir in range(n_radial + 1):
            r = r_min + ir * dr
            for iphi in range(n_phi):
                phi = iphi * dphi
                for iz in range(n_z + 1):
                    z = -z_max + iz * dz

                    # 检查点是否在有效灵敏体积内
                    # (不在探测器的支撑结构中)
                    in_gap = True
                    for layer in self.layers:
                        if abs(r - layer.radius_mm) < 0.5 * dr:
                            in_gap = False
                            break

                    if in_gap and r > 0:
                        grid_points.append((r, phi, z))

        return grid_points

    # ============================================================
    # [1423_xyz_display] 3D 坐标归一化 → 击中点坐标标准化
    # ============================================================
    def normalize_hit_coordinates(self, hits_xyz):
        """
        基于 [1423_xyz_display] 的坐标归一化方法

        原算法计算各轴 min/max 后添加 2.5% 边距进行归一化:
            range = max(xyz) - min(xyz)
            margin = 0.025 * range
            x_norm = (x - min_x + margin) / (range + 2*margin)

        这里用于将探测器击中坐标归一化到 [0,1]³ 范围，
        作为模式识别的预处理步骤

        Parameters
        ----------
        hits_xyz : list of tuple
            [(x, y, z), ...] 笛卡尔坐标击中列表

        Returns
        -------
        tuple : (normalized_hits, bounds_info)
        """
        if not hits_xyz:
            return [], {}

        # 计算各轴范围
        xs = [h[0] for h in hits_xyz]
        ys = [h[1] for h in hits_xyz]
        zs = [h[2] for h in hits_xyz]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        # 全局范围 (与 [1423] 一致)
        global_min = min(x_min, y_min, z_min)
        global_max = max(x_max, y_max, z_max)
        total_range = global_max - global_min

        if total_range < 1e-15:
            total_range = 1.0

        # 2.5% 边距 (继承自 [1423])
        margin = 0.025 * total_range

        normalized = []
        for (x, y, z) in hits_xyz:
            xn = (x - global_min + margin) / (total_range + 2.0 * margin)
            yn = (y - global_min + margin) / (total_range + 2.0 * margin)
            zn = (z - global_min + margin) / (total_range + 2.0 * margin)
            normalized.append((xn, yn, zn))

        bounds = {
            'global_min': global_min,
            'global_max': global_max,
            'range': total_range,
            'margin': margin,
        }

        return normalized, bounds

    # ============================================================
    # [1322_triangle_to_xml] 网格序列化 → 探测器几何 XML 导出
    # ============================================================
    def to_xml_string(self):
        """
        基于 [1322_triangle_to_xml] 的 DOLFIN XML 序列化格式

        原算法将 TRIANGLE 网格的节点和三角形单元写入 DOLFIN XML:
            <dolfin xmlns:dolfin="https://fenicsproject.org/">
              <mesh celltype="triangle" dim="2">
                <vertices size="N">
                  <vertex index="i" x="..." y="..."/>
                </vertices>
                <cells size="M">
                  <triangle index="j" v0="..." v1="..." v2="..."/>
                </cells>
              </mesh>
            </dolfin>

        这里将探测器层几何导出为类似格式:
        - 顶点 = 层的横截面控制点
        - 单元 = 层间的连接关系
        - 属性 = 材料参数

        Returns
        -------
        str : XML 格式字符串
        """
        lines = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append('<dolfin xmlns:dolfin="https://fenicsproject.org/">')
        lines.append('  <mesh celltype="cylinder_layer" dim="3">')

        # 顶点: 每层两个端点 (±z)
        n_vertices = 2 * self.n_layers
        lines.append(f'    <vertices size="{n_vertices}">')
        for i, layer in enumerate(self.layers):
            # 底端
            lines.append(
                f'      <vertex index="{2*i}" '
                f'x="{layer.radius_mm:.6f}" '
                f'y="0.000000" '
                f'z="{-layer.half_length_mm:.6f}" '
                f'thickness_ratio="{layer.thickness_ratio:.6e}" '
                f'resolution_r="{layer.sigma_r_mm:.6e}"/>'
            )
            # 顶端
            lines.append(
                f'      <vertex index="{2*i+1}" '
                f'x="{layer.radius_mm:.6f}" '
                f'y="0.000000" '
                f'z="{layer.half_length_mm:.6f}" '
                f'thickness_ratio="{layer.thickness_ratio:.6e}" '
                f'resolution_z="{layer.sigma_z_mm:.6e}"/>'
            )
        lines.append('    </vertices>')

        # 单元: 层间连接
        n_cells = max(self.n_layers - 1, 0)
        lines.append(f'    <cells size="{n_cells}">')
        for i in range(n_cells):
            lines.append(
                f'      <cell index="{i}" '
                f'v0="{2*i}" v1="{2*i+1}" '
                f'v2="{2*(i+1)}" v3="{2*(i+1)+1}" '
                f'material="{self.layers[i].material}"/>'
            )
        lines.append('    </cells>')
        lines.append('  </mesh>')
        lines.append('</dolfin>')

        return '\n'.join(lines)

    # ============================================================
    # [1330_triangulation] 网格质量评估 → 探测器覆盖质量
    # ============================================================
    def coverage_quality_score(self, n_phi_samples=36):
        """
        基于 [1330_triangulation] 的网格质量度量思想

        原算法计算三角网格的质量:
            q = 3.4641 · |A| / (a² + b² + c²)
        其中 A 为面积，a,b,c 为边长

        这里用于评估探测器方位角覆盖的均匀性:
        将 φ 空间分为 n_phi_samples 个扇区，
        检查每层在各扇区中是否有均匀覆盖

        Returns
        -------
        float : 质量分数 ∈ [0, 1]，1 为完美均匀覆盖
        """
        dphi = 2.0 * math.pi / n_phi_samples
        quality_scores = []

        for layer in self.layers:
            # 每层的理想覆盖率
            area_total = layer.surface_area_mm2()
            # 每个扇区的理想面积
            area_per_sector = area_total / n_phi_samples

            # 模拟每扇区的实际覆盖
            sector_areas = []
            for isec in range(n_phi_samples):
                phi_center = (isec + 0.5) * dphi
                # 扇区覆盖面积 (简化: 假设完美柱面)
                sector_area = (layer.radius_mm * dphi *
                               2.0 * layer.half_length_mm)
                sector_areas.append(sector_area)

            # 计算均匀性 (类似 [1330] 的 q_measure)
            mean_area = sum(sector_areas) / len(sector_areas)
            variance = sum((a - mean_area)**2 for a in sector_areas) / len(sector_areas)
            std_area = math.sqrt(variance) if variance > 0 else 0.0

            # 质量分数 = 1 - CV (变异系数)
            cv = std_area / mean_area if mean_area > 1e-15 else 1.0
            quality = max(0.0, 1.0 - cv)
            quality_scores.append(quality)

        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # ============================================================
    # [503_hand_mesh2d] 网格细化 → 探测器层密度优化
    # ============================================================
    def refine_layer_density(self, target_max_spacing_mm):
        """
        基于 [503_hand_mesh2d] 的 DistMesh 迭代细化思想

        原算法通过以下步骤生成自适应网格:
        1. Quadtree 分解估计局部特征尺寸 h(x,y)
        2. 约束 Delaunay 三角化
        3. 弹簧力平滑
        4. 在低质量区域添加新节点
        5. 迭代至收敛

        这里简化为: 根据目标间距在各层上均匀插入附加采样点，
        用于评估探测器是否需要更精细的读出分割

        Parameters
        ----------
        target_max_spacing_mm : float
            目标最大间距 [mm]

        Returns
        -------
        dict : 每层所需的读出通道数
        """
        channel_counts = {}
        for layer in self.layers:
            circumference = 2.0 * math.pi * layer.radius_mm
            axial_length = 2.0 * layer.half_length_mm

            # 方位角方向通道数 (类似 [503] 的 h(x,y) 控制)
            n_phi = max(1, int(math.ceil(circumference / target_max_spacing_mm)))
            # 纵向通道数
            n_z = max(1, int(math.ceil(axial_length / target_max_spacing_mm)))

            channel_counts[layer.layer_id] = {
                'n_phi': n_phi,
                'n_z': n_z,
                'total_channels': n_phi * n_z,
                'actual_spacing_phi': circumference / n_phi,
                'actual_spacing_z': axial_length / n_z,
            }

        return channel_counts

    def summary(self):
        """生成探测器摘要信息"""
        lines = []
        lines.append(f"TrackingDetector: {self.n_layers} layers")
        lines.append(f"  Radius range: [{self.inner_radius:.1f}, {self.outer_radius:.1f}] mm")
        lines.append(f"  Total material budget: {self._total_material_budget:.4f} X₀")
        for layer in self.layers:
            lines.append(f"  Layer {layer.layer_id}: r={layer.radius_mm:.1f}mm, "
                         f"t/X₀={layer.thickness_ratio:.4f}, "
                         f"σ_r={layer.sigma_r_mm*1e3:.1f}μm, "
                         f"σ_z={layer.sigma_z_mm*1e3:.1f}μm, "
                         f"mat={layer.material}")
        return '\n'.join(lines)


def build_standard_tracker():
    """
    构建标准 ATLAS/CMS 型硅微条追踪探测器

    基于典型 LHC 实验的 Inner Detector / Tracker 参数:
    - 4 层像素层 (内层)
    - 4 层硅微条层 (外层)
    - 总材料预算 ~0.4 X₀
    - 空间分辨率 ~10-100 μm
    """
    layers = []

    # 像素层 (4层)
    pixel_radii = [33.0, 50.5, 65.5, 88.0]  # mm
    pixel_thickness = [0.020, 0.020, 0.025, 0.025]  # X₀
    pixel_sigma_r = [0.010, 0.010, 0.010, 0.010]  # mm (10 μm)
    pixel_sigma_z = [0.115, 0.115, 0.115, 0.115]  # mm (115 μm)

    for i in range(4):
        layers.append(DetectorLayer(
            layer_id=i,
            radius_mm=pixel_radii[i],
            half_length_mm=250.0,
            thickness_ratio=pixel_thickness[i],
            sigma_r_mm=pixel_sigma_r[i],
            sigma_z_mm=pixel_sigma_z[i],
            material='silicon'
        ))

    # 硅微条层 (4层)
    strip_radii = [260.0, 360.0, 470.0, 580.0]  # mm
    strip_thickness = [0.030, 0.035, 0.040, 0.045]  # X₀
    strip_sigma_r = [0.029, 0.045, 0.075, 0.110]  # mm
    strip_sigma_z = [0.580, 0.580, 0.580, 0.580]  # mm

    for i in range(4):
        layers.append(DetectorLayer(
            layer_id=4 + i,
            radius_mm=strip_radii[i],
            half_length_mm=500.0,
            thickness_ratio=strip_thickness[i],
            sigma_r_mm=strip_sigma_r[i],
            sigma_z_mm=strip_sigma_z[i],
            material='silicon'
        ))

    return TrackingDetector(layers)
