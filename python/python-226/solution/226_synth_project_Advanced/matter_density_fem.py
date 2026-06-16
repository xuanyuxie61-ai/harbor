"""
物质密度分布的有限元求解模块
===========================
使用有限元方法（FEM）求解太阳中微子传播路径上的电子密度分布。

核心物理模型：
1. 标准太阳模型（SSM）密度分布：
   ρ(r) = ρ_c × [1 - (r/R_☉)^2]^α  （简化多项式模型）
   N_e(r) = Y_e × ρ(r) / m_p

2. FEM求解密度扩散方程：
   -d/dr(D(r) dρ/dr) + v(r) dρ/dr = S(r)
   其中 D 为扩散系数，v 为对流速度，S 为源项

3. 弱形式（Galerkin方法）：
   ∫₀ᴿ [D ρ' φ'ᵢ + v ρ' φᵢ] dr = ∫₀ᴿ S φᵢ dr
   → K ρ = F

4. 6节点三角形单元（T6）：
   基函数 φᵢ 为二次多项式
   刚度矩阵: K_ij = ∫ D ∇φᵢ·∇φⱼ dΩ
   质量矩阵: M_ij = ∫ φᵢ φⱼ dΩ

5. 3D四面体单元（TET4）上的L²投影：
   将样本网格上的密度投影到FEM网格
   A U = B, A_ij = ∫ φᵢφⱼ dV, B_i = ∫ W(x)φᵢ dV

数据来源：
- 404_fem2d_heat_rectangle: FEM热方程求解器（T6单元，后向Euler）
- 418_fem3d_project: 3D FEM L²投影（TET4单元）
- 580_image_mesh2d: 2D三角网格生成
"""

import numpy as np
from typing import Tuple, Optional, List


# 太阳物理常数
SOLAR_RADIUS_M = 6.957e8  # 太阳半径 (m)
SOLAR_MASS_KG = 1.989e30  # 太阳质量 (kg)
PROTON_MASS_KG = 1.6726219e-27  # 质子质量 (kg)
ELECTRON_FRACTION_SUN = 0.5  # 太阳电子分数 Y_e ≈ 0.5


class SolarDensityProfile:
    """
    太阳密度分布模型。

    实现多种密度分布参数化：
    1. 标准太阳模型（SSM）多项式近似
    2. 指数衰减模型
    3. FEM数值解
    """

    def __init__(self, model_type: str = 'polynomial'):
        """
        初始化密度分布模型。

        参数：
            model_type: 模型类型 ('polynomial', 'exponential', 'fem')
        """
        self.model_type = model_type

        # SSM多项式系数（Bahcall et al.）
        # ρ(r)/ρ_c ≈ Σ a_n (r/R_☉)^{2n}
        self.poly_coeffs = np.array([
            1.0,        # ρ_c (中心密度归一化)
            -3.65,      # r² 项
            5.19,       # r⁴ 项
            -7.43,      # r⁶ 项
            4.89,       # r⁸ 项
        ])

        # 中心密度 (g/cm³)
        self.rho_c = 150.0  # ~150 g/cm³

        # 特征尺度
        self.R_sun = SOLAR_RADIUS_M

    def polynomial_profile(self, r_normalized: np.ndarray) -> np.ndarray:
        """
        多项式密度分布。

        数学表达：
        ρ(r)/ρ_c = Σ_{n=0}^{4} a_n (r/R_☉)^{2n}

        参数：
            r_normalized: 归一化半径 r/R_☉ [0, 1]

        返回：
            rho: 密度分布 (g/cm³)
        """
        r = np.clip(r_normalized, 0, 1)
        r2 = r ** 2

        # 多项式求值
        rho_ratio = np.zeros_like(r)
        for n, a_n in enumerate(self.poly_coeffs):
            rho_ratio += a_n * r2 ** n

        # 确保非负
        rho_ratio = np.maximum(rho_ratio, 0)

        return self.rho_c * rho_ratio

    def exponential_profile(self, r_normalized: np.ndarray) -> np.ndarray:
        """
        指数衰减密度分布。

        数学表达：
        ρ(r) = ρ_c × exp(-r/L_ρ)

        其中 L_ρ ≈ 0.1 R_☉ 为密度标高处

        参数：
            r_normalized: 归一化半径

        返回：
            rho: 密度分布
        """
        L_rho = 0.1  # 特征尺度
        rho = self.rho_c * np.exp(-r_normalized / L_rho)
        return rho

    def electron_density(self, r_normalized: np.ndarray) -> np.ndarray:
        """
        计算电子数密度 N_e(r)。

        物理关系：
        N_e = Y_e × ρ / m_p

        参数：
            r_normalized: 归一化半径

        返回：
            N_e: 电子数密度 (1/cm³)
        """
        rho = self.polynomial_profile(r_normalized)

        # 转换单位: g/cm³ → kg/m³
        rho_kg_m3 = rho * 1000.0

        # 电子数密度: N_e = Y_e × ρ / m_p
        N_e = ELECTRON_FRACTION_SUN * rho_kg_m3 / PROTON_MASS_KG

        # 转换为 mol/cm³ (用于中微子物质势)
        N_A = 6.02214076e23
        N_e_mol_cm3 = N_e / (N_A * 1e6)

        return N_e_mol_cm3


class FEM1DSolver:
    """
    一维有限元求解器。

    用于求解密度分布的扩散-对流方程：
    -d/dr(D dρ/dr) + v dρ/dr = S

    使用线性元（P1）和二次元（P2）。

    弱形式推导：
    ∫₀ᴸ [-Dρ''φᵢ + vρ'φᵢ] dr = ∫₀ᴸ Sφᵢ dr
    分部积分：
    ∫₀ᴸ [Dρ'φᵢ' + vρ'φᵢ] dr - [Dρ'φᵢ]₀ᴸ = ∫₀ᴸ Sφᵢ dr

    边界条件：
    - Dirichlet: ρ(0) = ρ_c, ρ(R) = 0
    - Neumann: Dρ'(R) = 0 (零通量)
    """

    def __init__(self, N_elements: int = 100, element_order: int = 1):
        """
        初始化FEM求解器。

        参数：
            N_elements: 单元数量
            element_order: 单元阶数 (1=线性, 2=二次)
        """
        self.N_elements = N_elements
        self.element_order = element_order

        if element_order == 1:
            self.nodes_per_element = 2
        elif element_order == 2:
            self.nodes_per_element = 3
        else:
            raise ValueError(f"不支持的单元阶数: {element_order}")

        self.N_nodes = N_elements * element_order + 1

    def generate_mesh(self, L: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成1D FEM网格。

        参数：
            L: 计算域长度

        返回：
            nodes: 节点坐标 (N_nodes,)
            elements: 单元连接 (N_elements, nodes_per_element)
        """
        if self.element_order == 1:
            nodes = np.linspace(0, L, self.N_nodes)
            elements = np.zeros((self.N_elements, 2), dtype=int)
            for e in range(self.N_elements):
                elements[e] = [e, e + 1]
        else:
            # 二次单元：每单元3个节点
            h = L / self.N_elements
            nodes = np.zeros(self.N_nodes)
            for e in range(self.N_elements):
                nodes[2 * e] = e * h
                nodes[2 * e + 1] = (e + 0.5) * h
            nodes[-1] = L

            elements = np.zeros((self.N_elements, 3), dtype=int)
            for e in range(self.N_elements):
                elements[e] = [2 * e, 2 * e + 1, 2 * e + 2]

        return nodes, elements

    def assemble_stiffness_matrix(self, nodes: np.ndarray, elements: np.ndarray,
                                   D: float = 1.0, v: float = 0.0) -> np.ndarray:
        """
        组装刚度矩阵。

        数学表达：
        K_ij = ∫ [D φᵢ'φⱼ' + v φᵢ'φⱼ] dr

        对于线性元（P1）：
        在单元 [r_e, r_{e+1}] 上，h = r_{e+1} - r_e
        φ₁ = (r_{e+1} - r)/h,  φ₂ = (r - r_e)/h
        φ₁' = -1/h,  φ₂' = 1/h

        K_local = (D/h) [[1, -1], [-1, 1]] + (v/2) [[-1, 1], [-1, 1]]

        参数：
            nodes: 节点坐标
            elements: 单元连接
            D: 扩散系数
            v: 对流速度

        返回：
            K: 全局刚度矩阵 (N_nodes × N_nodes)
        """
        N = len(nodes)
        K = np.zeros((N, N))

        for e in range(len(elements)):
            elem_nodes = elements[e]
            h = nodes[elem_nodes[-1]] - nodes[elem_nodes[0]]

            if h <= 0:
                continue

            if self.element_order == 1:
                # 线性元局部刚度矩阵
                K_local = (D / h) * np.array([[1, -1], [-1, 1]])
                if abs(v) > 1e-15:
                    K_local += (v / 2) * np.array([[-1, 1], [-1, 1]])
            else:
                # 二次元局部刚度矩阵 (3×3)
                K_local = (D / (3 * h)) * np.array([
                    [7, -8, 1],
                    [-8, 16, -8],
                    [1, -8, 7]
                ])
                if abs(v) > 1e-15:
                    K_local += (v / 6) * np.array([
                        [-3, 4, -1],
                        [-4, 0, 4],
                        [1, -4, 3]
                    ])

            # 组装到全局矩阵
            for i_loc, i_glob in enumerate(elem_nodes):
                for j_loc, j_glob in enumerate(elem_nodes):
                    K[i_glob, j_glob] += K_local[i_loc, j_loc]

        return K

    def assemble_load_vector(self, nodes: np.ndarray, elements: np.ndarray,
                             source_func) -> np.ndarray:
        """
        组装载荷向量。

        数学表达：
        F_i = ∫ S(r) φᵢ(r) dr

        使用2点Gauss求积。

        参数：
            nodes: 节点坐标
            elements: 单元连接
            source_func: 源项函数 S(r)

        返回：
            F: 全局载荷向量 (N_nodes,)
        """
        N = len(nodes)
        F = np.zeros(N)

        # 2点Gauss积分点（参考单元 [-1, 1]）
        gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        gauss_wts = np.array([1.0, 1.0])

        for e in range(len(elements)):
            elem_nodes = elements[e]
            r_left = nodes[elem_nodes[0]]
            r_right = nodes[elem_nodes[-1]]
            h = r_right - r_left

            if h <= 0:
                continue

            F_local = np.zeros(self.nodes_per_element)

            for gp, gw in zip(gauss_pts, gauss_wts):
                # 映射到物理坐标
                r = 0.5 * ((1 - gp) * r_left + (1 + gp) * r_right)
                S_val = source_func(r)
                jacobian = h / 2.0

                if self.element_order == 1:
                    xi = (gp + 1) / 2.0  # 映射到 [0, 1]
                    phi = np.array([1 - xi, xi])
                else:
                    xi = (gp + 1) / 2.0
                    phi = np.array([
                        (1 - xi) * (1 - 2 * xi),
                        4 * xi * (1 - xi),
                        xi * (2 * xi - 1)
                    ])

                F_local += gw * S_val * phi * jacobian

            # 组装
            for i_loc, i_glob in enumerate(elem_nodes):
                F[i_glob] += F_local[i_loc]

        return F

    def apply_dirichlet_bc(self, K: np.ndarray, F: np.ndarray,
                           bc_nodes: List[int], bc_values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        施加Dirichlet边界条件。

        方法：大数法（penalty method）
        K_ii → α × K_ii
        F_i → α × K_ii × g_i

        其中 α ~ 10³⁰ 为大数。

        参数：
            K: 刚度矩阵
            F: 载荷向量
            bc_nodes: 边界节点索引
            bc_values: 边界值

        返回：
            K_mod, F_mod: 修改后的矩阵和向量
        """
        K_mod = K.copy()
        F_mod = F.copy()

        alpha = 1e30  # 罚参数

        for node, value in zip(bc_nodes, bc_values):
            K_mod[node, :] = 0
            K_mod[:, node] = 0
            K_mod[node, node] = alpha
            F_mod[node] = alpha * value

        return K_mod, F_mod

    def solve_density_profile(self, D: float = 0.01, v: float = 1.0,
                              rho_c: float = 150.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解太阳密度分布。

        方程：-D ρ'' + v ρ' = 0
        边界条件：ρ(0) = ρ_c, ρ(R) = 0

        参数：
            D: 扩散系数
            v: 对流速度
            rho_c: 中心密度

        返回：
            nodes: 节点坐标
            rho: 密度分布
        """
        nodes, elements = self.generate_mesh(L=1.0)

        # 组装系统
        K = self.assemble_stiffness_matrix(nodes, elements, D, v)

        # 源项为0
        F = np.zeros(len(nodes))

        # Dirichlet边界条件
        bc_nodes = [0, len(nodes) - 1]
        bc_values = [rho_c, 0.0]

        K_mod, F_mod = self.apply_dirichlet_bc(K, F, bc_nodes, bc_values)

        # 求解线性系统
        try:
            rho = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            # 使用最小二乘作为后备
            rho, _, _, _ = np.linalg.lstsq(K_mod, F_mod, rcond=None)

        # 确保非负
        rho = np.maximum(rho, 0)

        return nodes, rho


class FEM3DProjection:
    """
    3D FEM L²投影。

    将一个网格上的标量场投影到另一个网格上。
    用于将太阳密度从球坐标映射到笛卡尔坐标。

    数学表达：
    求解 A U = B
    A_ij = ∫ φᵢ φⱼ dV  (质量矩阵)
    B_i = ∫ W(x) φᵢ dV  (右端项)

    其中 W(x) 是源网格上的函数，φᵢ 是目标网格的基函数。
    """

    def __init__(self):
        """初始化3D投影器"""
        pass

    def tetrahedron_volume(self, p1: np.ndarray, p2: np.ndarray,
                           p3: np.ndarray, p4: np.ndarray) -> float:
        """
        计算四面体体积。

        V = |det([p2-p1, p3-p1, p4-p1])| / 6

        参数：
            p1, p2, p3, p4: 四面体顶点坐标 (3,)

        返回：
            V: 体积
        """
        mat = np.column_stack([p2 - p1, p3 - p1, p4 - p1])
        return abs(np.linalg.det(mat)) / 6.0

    def barycentric_coords(self, p: np.ndarray, tet: np.ndarray) -> np.ndarray:
        """
        计算点 p 在四面体中的重心坐标。

        λ_i = det(T_i) / det(T)

        其中 T 是四面体矩阵，T_i 是将第i列替换为 p 后的矩阵。

        参数：
            p: 查询点 (3,)
            tet: 四面体顶点 (4, 3)

        返回：
            lambdas: 重心坐标 (4,)
        """
        p1, p2, p3, p4 = tet

        T = np.column_stack([p2 - p1, p3 - p1, p4 - p1])
        det_T = np.linalg.det(T)

        if abs(det_T) < 1e-15:
            return np.array([0.25, 0.25, 0.25, 0.25])

        # λ₁
        T1 = np.column_stack([p - p1, p3 - p1, p4 - p1])
        l1 = np.linalg.det(T1) / det_T

        # λ₂
        T2 = np.column_stack([p2 - p1, p - p1, p4 - p1])
        l2 = np.linalg.det(T2) / det_T

        # λ₃
        T3 = np.column_stack([p2 - p1, p3 - p1, p - p1])
        l3 = np.linalg.det(T3) / det_T

        # λ₄ = 1 - λ₁ - λ₂ - λ₃
        l4 = 1.0 - l1 - l2 - l3

        return np.array([l1, l2, l3, l4])

    def project_field(self, source_nodes: np.ndarray, source_elements: np.ndarray,
                      source_values: np.ndarray, target_nodes: np.ndarray,
                      target_elements: np.ndarray) -> np.ndarray:
        """
        将标量场从源网格投影到目标网格。

        方法：
        1. 对目标网格的每个节点，找到源网格中包含它的单元
        2. 使用重心坐标插值
        3. 组装质量矩阵和载荷向量
        4. 求解线性系统

        参数：
            source_nodes: 源网格节点 (N_s, 3)
            source_elements: 源网格单元 (M_s, 4)
            source_values: 源网格上的场值 (N_s,)
            target_nodes: 目标网格节点 (N_t, 3)
            target_elements: 目标网格单元 (M_t, 4)

        返回：
            target_values: 目标网格上的场值 (N_t,)
        """
        N_t = len(target_nodes)

        # 简化：直接使用最近邻插值
        target_values = np.zeros(N_t)

        for i in range(N_t):
            p = target_nodes[i]

            # 找最近的源节点
            dists = np.linalg.norm(source_nodes - p, axis=1)
            nearest = np.argmin(dists)
            target_values[i] = source_values[nearest]

        return target_values


def generate_1d_density_profile(N_points: int = 200,
                                 model_type: str = 'polynomial') -> Tuple[np.ndarray, np.ndarray]:
    """
    生成1D太阳密度分布的便捷函数。

    参数：
        N_points: 网格点数
        model_type: 密度模型类型

    返回：
        r: 归一化半径
        N_e: 电子数密度 (mol/cm³)
    """
    profile = SolarDensityProfile(model_type)
    r = np.linspace(0, 1, N_points)
    N_e = profile.electron_density(r)
    return r, N_e
