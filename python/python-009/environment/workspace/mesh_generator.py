"""
mesh_generator.py
行星大气数值网格生成模块。

融合原始项目：308_distmesh（基于距离函数的网格生成）
             1320_triangle_to_fem（网格格式转换）

在系外行星大气辐射传输计算中，需要在(r, θ)或(r, μ)空间中
生成高质量的计算网格。本模块实现：
- 一维自适应压强分层
- 二维球壳截面网格生成
- 网格质量评估与优化
"""

import numpy as np
from typing import Tuple, List, Optional


class AtmosphericMesh:
    """
    行星大气一维/二维计算网格。

    一维垂直网格:
        按压强分层，每层有中心压强 P_i、界面压强 P_{i+1/2}
        层厚度 Δz_i = z_{i+1/2} - z_{i-1/2}

    二维轴对称网格 (r, μ):
        在径向-角度空间生成结构化网格，用于有限元离散化。
    """

    def __init__(self, n_layers: int, P_top: float, P_bot: float,
                 planet_radius_m: float):
        self.n_layers = n_layers
        self.P_top = P_top
        self.P_bot = P_bot
        self.R_p = planet_radius_m

        # 生成对数等间距压强节点
        self.P_interface = self._log_pressure_grid(n_layers + 1, P_top, P_bot)
        self.P_center = 0.5 * (self.P_interface[:-1] + self.P_interface[1:])
        self.dP = np.diff(self.P_interface)

    def _log_pressure_grid(self, n: int, P_min: float, P_max: float) -> np.ndarray:
        """对数等间距压强网格。"""
        logP = np.linspace(np.log10(P_min), np.log10(P_max), n)
        return 10.0**logP

    def adaptive_refinement(self, error_estimator: np.ndarray,
                            max_layers: int = 200,
                            tol: float = 0.1) -> "AtmosphericMesh":
        """
        基于误差估计的自适应网格加密。

        融合 distmesh 的自适应思想：根据局部误差指示器动态调整网格分辨率。

        算法:
            1. 计算每层相对误差 η_i = error_i / max(error)
            2. 若 η_i > tol，将该层一分为二
            3. 重复直到所有 η_i ≤ tol 或达到最大层数

        参数:
            error_estimator: 每层误差估计，形状 (n_layers,)
            max_layers: 最大允许层数
            tol: 相对误差阈值

        返回:
            加密后的新网格
        """
        error_estimator = np.asarray(error_estimator, dtype=np.float64)
        if error_estimator.shape[0] != self.n_layers:
            raise ValueError("误差估计维度与层数不匹配")

        P_int = self.P_interface.copy()
        err = error_estimator.copy()

        while len(P_int) - 1 < max_layers:
            err_max = np.max(err)
            if err_max < 1e-30:
                break

            rel_err = err / err_max
            refine_mask = rel_err > tol

            if not np.any(refine_mask):
                break

            # 找到需要细化的层
            refine_idx = np.where(refine_mask)[0]
            if len(refine_idx) == 0:
                break

            new_P_int = [P_int[0]]
            new_err = []
            for i in range(len(P_int) - 1):
                new_P_int.append(P_int[i + 1])
                new_err.append(err[i])
                if i in refine_idx and len(new_P_int) < max_layers + 1:
                    # 在层中间插入新界面
                    P_mid = np.sqrt(P_int[i] * P_int[i + 1])
                    new_P_int.insert(-1, P_mid)
                    # 新层误差估计：假设误差按 h^2 收敛，减半后误差为原 1/4
                    new_err.insert(-1, err[i] * 0.25)

            P_int = np.array(new_P_int)
            err = np.array(new_err)

        mesh = AtmosphericMesh.__new__(AtmosphericMesh)
        mesh.n_layers = len(P_int) - 1
        mesh.P_top = P_int[0]
        mesh.P_bot = P_int[-1]
        mesh.R_p = self.R_p
        mesh.P_interface = P_int
        mesh.P_center = 0.5 * (P_int[:-1] + P_int[1:])
        mesh.dP = np.diff(P_int)
        return mesh

    def generate_2d_shell_mesh(self, n_angular: int = 32,
                                r_min_factor: float = 1.0,
                                r_max_factor: float = 1.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成二维球壳截面结构化网格 (r, θ)。

        融合 distmesh 的距离函数思想：在大气层内生成非均匀径向网格，
        靠近行星表面处加密。

        网格节点坐标:
            r_j = R_p * [1 + (r_max_factor - 1) * ξ_j]
            θ_i = i * π / (n_angular - 1)

        其中 ξ_j 在 [0, 1] 上按压强对数分布加密。

        返回:
            nodes: 节点坐标数组，形状 (n_nodes, 2)，每行 (r, θ)
            elements: 三角形单元连接，形状 (n_elements, 3)，0-based
        """
        if n_angular < 3:
            raise ValueError("角度方向节点数至少为 3")

        # 径向节点数与压强层数一致
        n_radial = self.n_layers + 1

        # 径向坐标：在近表面加密
        xi = np.linspace(0.0, 1.0, n_radial)
        # 使用距离函数的变形：靠近边界加密
        xi = xi**1.5  # 幂律加密
        r_nodes = self.R_p * (r_min_factor + (r_max_factor - r_min_factor) * xi)

        theta_nodes = np.linspace(0.0, np.pi, n_angular)

        # 构建节点列表
        nodes = []
        for j in range(n_radial):
            for i in range(n_angular):
                nodes.append([r_nodes[j], theta_nodes[i]])
        nodes = np.array(nodes, dtype=np.float64)

        # 构建三角形单元（结构化四边形剖分为两个三角形）
        elements = []
        for j in range(n_radial - 1):
            for i in range(n_angular - 1):
                n0 = j * n_angular + i
                n1 = j * n_angular + (i + 1)
                n2 = (j + 1) * n_angular + i
                n3 = (j + 1) * n_angular + (i + 1)
                # 两个三角形
                elements.append([n0, n1, n2])
                elements.append([n1, n3, n2])
        elements = np.array(elements, dtype=np.int64)

        return nodes, elements

    def mesh_quality_metrics(self, nodes: np.ndarray, elements: np.ndarray) -> dict:
        """
        评估三角形网格质量。

        质量指标:
            - 最小角: θ_min
            - 最大角: θ_max
            - 纵横比: 最长边 / 最短边
            - 面积变化系数

        融合 distmesh 中 mesh quality 评估思想。
        """
        if elements.shape[1] != 3:
            raise ValueError("仅支持三角形单元")

        qualities = []
        min_angles = []
        max_angles = []
        aspect_ratios = []

        for tri in elements:
            p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]

            # 边长
            a = np.linalg.norm(p1 - p2)
            b = np.linalg.norm(p0 - p2)
            c = np.linalg.norm(p0 - p1)

            # 避免退化
            if a < 1e-15 or b < 1e-15 or c < 1e-15:
                continue

            # 用余弦定理计算角度
            ang0 = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0))
            ang1 = np.arccos(np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0))
            ang2 = np.arccos(np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0))

            angles = np.array([ang0, ang1, ang2]) * 180.0 / np.pi
            min_angles.append(np.min(angles))
            max_angles.append(np.max(angles))
            aspect_ratios.append(np.max([a, b, c]) / np.min([a, b, c]))

            # 质量因子: 内切圆半径 / 外接圆半径 * 2
            s = 0.5 * (a + b + c)
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-30))
            r_in = area / s
            r_circ = a * b * c / (4.0 * max(area, 1e-30))
            quality = 2.0 * r_in / max(r_circ, 1e-30)
            qualities.append(quality)

        return {
            'min_angle_deg': np.min(min_angles) if min_angles else 0.0,
            'max_angle_deg': np.max(max_angles) if max_angles else 180.0,
            'mean_quality': np.mean(qualities) if qualities else 0.0,
            'min_quality': np.min(qualities) if qualities else 0.0,
            'max_aspect_ratio': np.max(aspect_ratios) if aspect_ratios else 1e10
        }


def distance_function_sphere_shell(points: np.ndarray, R_inner: float,
                                    R_outer: float) -> np.ndarray:
    """
    球壳区域的符号距离函数。

    融合 distmesh 的核心思想：用符号距离函数描述计算域。

    定义:
        d(r) = max(R_inner - r, r - R_outer)

    性质:
        d < 0: 点在区域内
        d = 0: 点在边界上
        d > 0: 点在区域外

    参数:
        points: 点坐标数组，形状 (N, 2) 或 (N, 3)
        R_inner: 内半径
        R_outer: 外半径

    返回:
        距离值数组
    """
    points = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(points, axis=1)
    d_inner = R_inner - r
    d_outer = r - R_outer
    return np.maximum(d_inner, d_outer)


def mesh_size_function(points: np.ndarray, R_p: float,
                        h_min: float = 1e3, h_max: float = 1e5) -> np.ndarray:
    """
    网格尺寸控制函数。

    在大气底层（靠近 R_p）加密，在高层稀疏：
        h(r) = h_min + (h_max - h_min) * [(r - R_p) / (H_atm)]^α

    参数:
        points: 点坐标
        R_p: 行星半径
        h_min: 最小网格尺寸
        h_max: 最大网格尺寸

    返回:
        各点期望网格尺寸
    """
    points = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(points, axis=1)
    delta_r = np.maximum(r - R_p, 0.0)
    H_atm = 5e6  # 典型大气标高 (m)
    alpha = 0.8
    h = h_min + (h_max - h_min) * (delta_r / H_atm)**alpha
    return np.clip(h, h_min, h_max)
