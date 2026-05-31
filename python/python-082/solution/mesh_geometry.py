# -*- coding: utf-8 -*-
"""
mesh_geometry.py
================
复合材料层合板几何建模与一维网格生成模块。

源自种子项目 891_polygonal_surface_display 的网格数据结构思想，
去除全部可视化代码，专注于计算几何描述。

科学背景：
---------
纤维增强复合材料层合板（Fiber-Reinforced Polymer Laminate, FRP）
由多层单向预浸料（ply）按特定铺层角堆叠而成。本模块为一维杆件模型
生成非均匀计算网格，在材料界面、潜在损伤区域和边界处进行局部加密。

核心公式：
  层合板等效厚度：
    h_total = sum_{k=1}^{N_ply} h_k
  第 k 层的中面坐标（厚度方向）：
    z_k = -h_total/2 + sum_{j=1}^{k-1} h_j + h_k/2
  非均匀网格映射（几何加密）：
    x(ξ) = L * (ξ + a*sin(π*ξ)) / (1 + a*sin(π)) , ξ∈[0,1]
  其中 a 控制加密强度，在 ξ=0.5 处网格最密（模拟冲击损伤区）。
"""

import numpy as np
from typing import List, Tuple, Optional


class CompositePly:
    """
    单向复合材料铺层（ply）属性定义。
    """

    def __init__(self, thickness: float, fiber_angle: float,
                 E1: float, E2: float, G12: float, nu12: float,
                 Vf: float, density: float):
        """
        Parameters
        ----------
        thickness : float
            单层厚度 h_k [m].
        fiber_angle : float
            纤维方向角 θ [deg]，相对于全局 x 轴。
        E1 : float
            纵向（纤维方向）弹性模量 [Pa].
        E2 : float
            横向（垂直纤维方向）弹性模量 [Pa].
        G12 : float
            面内剪切模量 [Pa].
        nu12 : float
            主泊松比。
        Vf : float
            纤维体积分数 [0,1].
        density : float
            材料密度 ρ [kg/m³].
        """
        if thickness <= 0:
            raise ValueError("Ply thickness must be positive.")
        if not (0.0 <= Vf <= 1.0):
            raise ValueError("Fiber volume fraction Vf must be in [0,1].")
        if E1 <= 0 or E2 <= 0 or G12 <= 0:
            raise ValueError("Elastic moduli must be positive.")

        self.thickness = thickness
        self.fiber_angle = np.radians(fiber_angle)
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.nu12 = nu12
        self.Vf = Vf
        self.density = density

        # 单层板转换刚度矩阵 Q_bar（平面应力）
        # 先计算材料主轴方向的折减刚度 Q
        nu21 = nu12 * E2 / E1
        denom = 1.0 - nu12 * nu21
        if abs(denom) < 1e-15:
            raise ValueError("Invalid Poisson ratio combination leads to singularity.")

        Q11 = E1 / denom
        Q12 = nu12 * E2 / denom
        Q22 = E2 / denom
        Q66 = G12

        c = np.cos(self.fiber_angle)
        s = np.sin(self.fiber_angle)
        c2 = c * c
        s2 = s * s
        c4 = c2 * c2
        s4 = s2 * s2

        # 转换到全局坐标系（经典层合板理论 CLT）
        self.Qbar = np.zeros((3, 3))
        self.Qbar[0, 0] = Q11 * c4 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * s4
        self.Qbar[0, 1] = (Q11 + Q22 - 4 * Q66) * s2 * c2 + Q12 * (s4 + c4)
        self.Qbar[1, 0] = self.Qbar[0, 1]
        self.Qbar[1, 1] = Q11 * s4 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * c4
        self.Qbar[0, 2] = (Q11 - Q12 - 2 * Q66) * c * s * c2 + (Q12 - Q22 + 2 * Q66) * c * s * s2
        self.Qbar[2, 0] = self.Qbar[0, 2]
        self.Qbar[1, 2] = (Q11 - Q12 - 2 * Q66) * c * s * s2 + (Q12 - Q22 + 2 * Q66) * c * s * c2
        self.Qbar[2, 1] = self.Qbar[1, 2]
        self.Qbar[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s2 * c2 + Q66 * (s4 + c4)

        # 一维等效轴向刚度（简化模型，取 Qbar[0,0]）
        self.E_eff = self.Qbar[0, 0]


class CompositeLaminate:
    """
    复合材料层合板几何与材料属性聚合。
    """

    def __init__(self, plies: List[CompositePly]):
        if not plies:
            raise ValueError("At least one ply is required.")
        self.plies = plies
        self.num_plies = len(plies)
        self.total_thickness = sum(p.thickness for p in plies)

        # 计算各层中面坐标 z_k（厚度方向）
        self.z_mids = np.zeros(self.num_plies)
        z = -self.total_thickness / 2.0
        for i, p in enumerate(plies):
            z += p.thickness / 2.0
            self.z_mids[i] = z
            z += p.thickness / 2.0

        # 等效密度（按体积加权平均）
        self.rho_eff = sum(p.density * p.thickness for p in plies) / self.total_thickness
        # 等效轴向刚度（按厚度加权平均，简化一维模型）
        self.E_eff = sum(p.E_eff * p.thickness for p in plies) / self.total_thickness

    def get_homogenized_properties(self) -> dict:
        """返回等效均质化材料属性字典。"""
        return {
            "rho_eff": self.rho_eff,
            "E_eff": self.E_eff,
            "h_total": self.total_thickness,
            "num_plies": self.num_plies,
        }


class Mesh1D:
    """
    一维非均匀计算网格生成器。

    网格映射公式（正弦加密映射）：
      x_i = L * (ξ_i + a * sin(π * ξ_i)) / (1 + a * sin(π))
    其中 ξ_i 为均匀分布的参数节点，a >= 0 控制加密强度。
    当 a=0 时为均匀网格；a>0 时中部加密（模拟冲击损伤核心区）。
    """

    def __init__(self, x_min: float, x_max: float, num_elements: int,
                 refine_strength: float = 0.0, refine_center: Optional[float] = None):
        """
        Parameters
        ----------
        x_min, x_max : float
            计算域边界 [m].
        num_elements : int
            单元总数（必须为 >= 2 的整数）。
        refine_strength : float
            加密强度 a（>=0）。
        refine_center : float or None
            加密中心位置；None 时取中点。
        """
        if num_elements < 2:
            raise ValueError("num_elements must be >= 2.")
        if x_max <= x_min:
            raise ValueError("x_max must be > x_min.")
        if refine_strength < 0:
            raise ValueError("refine_strength must be non-negative.")

        self.x_min = x_min
        self.x_max = x_max
        self.L = x_max - x_min
        self.num_elements = num_elements
        self.refine_strength = refine_strength
        self.refine_center = refine_center if refine_center is not None else (x_min + x_max) / 2.0

        # 生成节点
        self.nodes = self._generate_nodes()
        self.elements = self._build_elements()
        self.element_sizes = self.nodes[1:] - self.nodes[:-1]

        # 材料界面标记（预留，用于后续多材料问题）
        self.interface_flags = np.zeros(num_elements, dtype=bool)

    def _generate_nodes(self) -> np.ndarray:
        """生成非均匀分布的网格节点。

        映射公式（保证单调递增）：
          f(ξ) = ξ + a * (1 - cos(π*ξ)) / 2
          x = x_min + L * f(ξ) / f(1)
        导数 f'(ξ) = 1 + a*π*sin(π*ξ)/2 >= 1，始终单调。
        该映射在 ξ=0.5（x=L/2）处产生最大加密。
        """
        xi = np.linspace(0.0, 1.0, self.num_elements + 1)
        a = self.refine_strength
        # 映射函数及其在 ξ=1 的值（归一化用）
        f_xi = xi + a * 0.5 * (1.0 - np.cos(np.pi * xi))
        f_1 = 1.0 + a * 0.5 * (1.0 - np.cos(np.pi))  # = 1 + a
        x = self.x_min + self.L * f_xi / f_1
        # 数值鲁棒性：强制边界精确
        x[0] = self.x_min
        x[-1] = self.x_max
        # 确保单调递增
        if np.any(np.diff(x) <= 0):
            raise RuntimeError("Mesh generation failed: non-monotonic nodes detected.")
        return x

    def _build_elements(self) -> List[Tuple[int, int]]:
        """构建单元连通性（节点索引对）。"""
        elements = []
        for i in range(self.num_elements):
            elements.append((i, i + 1))
        return elements

    def get_element_centers(self) -> np.ndarray:
        """返回各单元中心坐标。"""
        return 0.5 * (self.nodes[:-1] + self.nodes[1:])

    def get_element_jacobians(self) -> np.ndarray:
        """
        返回各单元从参考单元 [-1,1] 到物理单元的 Jacobian。
        J_e = (x_{i+1} - x_i) / 2.
        """
        return self.element_sizes / 2.0

    def locate_point(self, x: float) -> int:
        """
        二分查找定位点 x 所在的单元索引。
        边界鲁棒性处理：x == x_max 时归入最后一个单元。
        """
        if x < self.x_min or x > self.x_max:
            raise ValueError(f"Point x={x} out of domain [{self.x_min}, {self.x_max}].")
        if x >= self.x_max:
            return self.num_elements - 1
        # 二分查找
        left, right = 0, self.num_elements
        while left < right:
            mid = (left + right) // 2
            if self.nodes[mid] <= x < self.nodes[mid + 1]:
                return mid
            elif x < self.nodes[mid]:
                right = mid
            else:
                left = mid + 1
        return min(left, self.num_elements - 1)


def build_default_laminate() -> CompositeLaminate:
    """
    构建一个标准 [0/45/-45/90]_s 对称铺层的碳纤维/环氧树脂层合板。
    材料参数参考 T300/5208 典型值。
    """
    # T300/5208 典型性能
    E1 = 181.0e9      # Pa
    E2 = 10.3e9       # Pa
    G12 = 7.17e9      # Pa
    nu12 = 0.28
    rho = 1600.0      # kg/m³
    Vf = 0.62
    h_ply = 0.125e-3  # 每层 0.125 mm

    angles = [0, 45, -45, 90, 90, -45, 45, 0]
    plies = []
    for theta in angles:
        plies.append(CompositePly(
            thickness=h_ply,
            fiber_angle=theta,
            E1=E1, E2=E2, G12=G12, nu12=nu12,
            Vf=Vf, density=rho
        ))
    return CompositeLaminate(plies)


def build_default_mesh(L: float = 1.0, num_elements: int = 40,
                       refine_strength: float = 1.5) -> Mesh1D:
    """生成默认一维计算网格。"""
    return Mesh1D(x_min=0.0, x_max=L, num_elements=num_elements,
                  refine_strength=refine_strength, refine_center=L / 2.0)


if __name__ == "__main__":
    # 自测试
    laminate = build_default_laminate()
    props = laminate.get_homogenized_properties()
    print("Laminate properties:", props)

    mesh = build_default_mesh()
    print("Mesh nodes (first 5):", mesh.nodes[:5])
    print("Element sizes (first 5):", mesh.element_sizes[:5])
    print("Element centers (first 5):", mesh.get_element_centers()[:5])
    print("Locate 0.5:", mesh.locate_point(0.5))
