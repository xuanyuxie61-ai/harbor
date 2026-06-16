"""
perp_diffusion.py - 垂直方向扩散方程的有限元求解器

本模块融合以下种子项目的核心算法：
  - 408_fem2d_poisson_rectangle → 二维Poisson方程的二次有限元方法

功能：
  1. 二维矩形域上的有限元组装（线性三角形元）
  2. 数值积分（3点高斯规则）
  3. 稀疏刚度矩阵与载荷向量组装
  4. Dirichlet边界条件施加
  5. 交叉场扩散方程求解
  6. 各向异性扩散张量处理
"""

import numpy as np
from typing import Dict, Tuple, Optional
from iterative_solver import solve_sparse_cg, construct_banded_system


# =============================================================================
# 参考单元上的基函数（线性三角形 T3）
# =============================================================================
def triangle_basis_values(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算参考三角形上的基函数值及其导数

    参考三角形: 顶点 (0,0), (1,0), (0,1)

    基函数:
      N1 = 1 - xi - eta
      N2 = xi
      N3 = eta

    导数:
      dN1/dxi = -1, dN1/deta = -1
      dN2/dxi = 1,  dN2/deta = 0
      dN3/dxi = 0,  dN3/deta = 1

    参数:
        xi, eta: 参考坐标
    返回:
        N: shape (3,) 基函数值
        dN: shape (3, 2) 基函数对(xi, eta)的导数
    """
    N = np.array([1.0 - xi - eta, xi, eta])
    dN = np.array([
        [-1.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    return N, dN


# =============================================================================
# 3点高斯积分规则（三角形）
# =============================================================================
def triangle_quadrature_3pt() -> Tuple[np.ndarray, np.ndarray]:
    """
    3点高斯积分规则（精度阶数2）

    积分点:
      (1/6, 1/6), (2/3, 1/6), (1/6, 2/3)
    权重:
      1/6, 1/6, 1/6

    返回:
        points: shape (3, 2) 积分点
        weights: shape (3,) 权重
    """
    points = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 6.0],
    ])
    weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    return points, weights


def triangle_quadrature_7pt() -> Tuple[np.ndarray, np.ndarray]:
    """
    7点高斯积分规则（精度阶数4）

    用于更高精度的数值积分
    """
    # 积分点
    a1 = 0.059715871789770
    b1 = 0.470142064105115
    a2 = 0.797426985353087
    b2 = 0.101286507323456

    points = np.array([
        [1.0 / 3.0, 1.0 / 3.0],
        [a1, b1],
        [b1, a1],
        [b1, b1],
        [a2, b2],
        [b2, a2],
        [b2, b2],
    ])
    weights = np.array([
        0.1125,
        0.06619707639425,
        0.06619707639425,
        0.06619707639425,
        0.06296959027241,
        0.06296959027241,
        0.06296959027241,
    ])
    return points, weights


# =============================================================================
# 单元刚度矩阵
# =============================================================================
def compute_element_stiffness(coords: np.ndarray,
                                D_tensor: np.ndarray) -> np.ndarray:
    """
    计算单元刚度矩阵

    K_e = integral_Omega_e (grad N_i)^T D (grad N_j) dA

    其中 D 是扩散张量:
      D = [D_xx  D_xy]
          [D_yx  D_yy]

    对于各向同性扩散: D = D_0 * I

    参数:
        coords: shape (3, 2) 三角形顶点坐标
        D_tensor: shape (2, 2) 扩散张量
    返回:
        K_e: shape (3, 3) 单元刚度矩阵
    """
    # 计算Jacobian: J = dx/d(xi)
    # x = x1 + (x2-x1)*xi + (x3-x1)*eta
    # y = y1 + (y2-y1)*xi + (y3-y1)*eta
    J = np.array([
        [coords[1, 0] - coords[0, 0], coords[2, 0] - coords[0, 0]],
        [coords[1, 1] - coords[0, 1], coords[2, 1] - coords[0, 1]],
    ])

    det_J = np.linalg.det(J)
    if abs(det_J) < 1e-30:
        return np.zeros((3, 3))

    J_inv = np.linalg.inv(J)

    # 全局坐标下的导数: dN/dx = J^{-T} * dN/dxi
    _, dN_ref = triangle_basis_values(0.0, 0.0)  # 导数是常数
    dN_global = dN_ref @ J_inv.T  # shape (3, 2)

    # 单元刚度矩阵: K = |J| * (dN/dx)^T D (dN/dx)
    # 因为线性元的导数是常数，所以积分简化为面积*被积函数
    K_e = abs(det_J) * (dN_global @ D_tensor @ dN_global.T)

    return K_e


def compute_element_load(coords: np.ndarray, source_func: callable) -> np.ndarray:
    """
    计算单元载荷向量

    f_e = integral_Omega_e N_i * S(x,y) dA

    使用3点高斯积分

    参数:
        coords: shape (3, 2) 三角形顶点坐标
        source_func: 源项函数 S(x, y)
    返回:
        f_e: shape (3,) 单元载荷向量
    """
    quad_pts, quad_wts = triangle_quadrature_3pt()

    # Jacobian
    J = np.array([
        [coords[1, 0] - coords[0, 0], coords[2, 0] - coords[0, 0]],
        [coords[1, 1] - coords[0, 1], coords[2, 1] - coords[0, 1]],
    ])
    det_J = np.linalg.det(J)

    f_e = np.zeros(3)
    for q in range(len(quad_pts)):
        xi, eta = quad_pts[q]
        N, _ = triangle_basis_values(xi, eta)

        # 物理坐标
        x = coords[0, 0] + (coords[1, 0] - coords[0, 0]) * xi + (coords[2, 0] - coords[0, 0]) * eta
        y = coords[0, 1] + (coords[1, 1] - coords[0, 1]) * xi + (coords[2, 1] - coords[0, 1]) * eta

        S = source_func(x, y)
        f_e += quad_wts[q] * abs(det_J) * N * S

    return f_e


# =============================================================================
# 全局组装
# =============================================================================
def assemble_global_system(nodes: np.ndarray, elements: np.ndarray,
                            D_tensor: np.ndarray,
                            source_func: Optional[callable] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    组装全局有限元系统 K*u = f

    参数:
        nodes: shape (n_nodes, 2) 节点坐标
        elements: shape (n_elements, 3) 单元连接表
        D_tensor: shape (2, 2) 扩散张量
        source_func: 源项函数
    返回:
        K: shape (n_nodes, n_nodes) 全局刚度矩阵（稠密）
        f: shape (n_nodes,) 全局载荷向量
    """
    n_nodes = len(nodes)
    K = np.zeros((n_nodes, n_nodes))
    f = np.zeros(n_nodes)

    for e in range(len(elements)):
        elem_nodes = elements[e]
        coords = nodes[elem_nodes]

        # 单元刚度矩阵
        K_e = compute_element_stiffness(coords, D_tensor)

        # 单元载荷向量
        if source_func is not None:
            f_e = compute_element_load(coords, source_func)
        else:
            f_e = np.zeros(3)

        # 组装到全局
        for i in range(3):
            for j in range(3):
                K[elem_nodes[i], elem_nodes[j]] += K_e[i, j]
            f[elem_nodes[i]] += f_e[i]

    return K, f


# =============================================================================
# Dirichlet边界条件
# =============================================================================
def apply_dirichlet_bc(K: np.ndarray, f: np.ndarray,
                         bc_nodes: np.ndarray, bc_values: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    施加Dirichlet边界条件

    方法：将边界节点对应的行修改为单位矩阵行
    并修正右端项

    参数:
        K: 全局刚度矩阵
        f: 全局载荷向量
        bc_nodes: 边界节点编号
        bc_values: 边界值
    返回:
        K_mod, f_mod: 修改后的系统
    """
    K_mod = K.copy()
    f_mod = f.copy()

    for i, node in enumerate(bc_nodes):
        val = bc_values[i]

        # 修正右端项
        f_mod -= K_mod[:, node] * val

        # 清零行和列
        K_mod[node, :] = 0.0
        K_mod[:, node] = 0.0
        K_mod[node, node] = 1.0
        f_mod[node] = val

    return K_mod, f_mod


# =============================================================================
# 二维扩散求解器类
# =============================================================================
class PerpendicularDiffusionSolver:
    """
    二维交叉场扩散方程求解器

    求解: -nabla . (D . nabla T) = Q

    其中 D 是各向异性扩散张量，Q 是加热源
    """

    def __init__(self, x_range: Tuple[float, float] = (0.0, 1.0),
                  y_range: Tuple[float, float] = (0.0, 1.0),
                  nx: int = 16, ny: int = 16):
        """
        参数:
            x_range, y_range: 计算域
            nx, ny: 网格点数
        """
        from divertor_geometry import generate_triangular_mesh, compute_vertex_to_element_map

        self.x_range = x_range
        self.y_range = y_range
        self.nx = nx
        self.ny = ny

        # 生成网格
        mesh_data = generate_triangular_mesh(
            x_range[0], x_range[1],
            y_range[0], y_range[1],
            nx, ny
        )
        self.nodes = mesh_data['nodes']
        self.elements = mesh_data['elements']
        self.n_nodes = mesh_data['n_nodes']
        self.n_elements = mesh_data['n_elements']

        # 计算VTOE映射
        self.vtoe_ptr, self.vtoe_list = compute_vertex_to_element_map(
            self.elements, self.n_nodes
        )

        # 解向量
        self.solution = np.zeros(self.n_nodes)

    def set_diffusion_tensor(self, D_xx: float = 1.0, D_yy: float = 1.0,
                               D_xy: float = 0.0):
        """设置扩散张量"""
        self.D_tensor = np.array([
            [D_xx, D_xy],
            [D_xy, D_yy],
        ])

    def solve(self, source_func: Optional[callable] = None,
               bc_type: str = 'homogeneous_dirichlet') -> np.ndarray:
        """
        求解扩散方程

        参数:
            source_func: 源项 S(x,y)
            bc_type: 边界条件类型
        返回:
            solution: 节点上的温度分布
        """
        # 组装全局系统
        K, f = assemble_global_system(
            self.nodes, self.elements,
            self.D_tensor, source_func
        )

        # 施加边界条件
        bc_nodes, bc_values = self._get_boundary_nodes(bc_type)
        K, f = apply_dirichlet_bc(K, f, bc_nodes, bc_values)

        # 求解线性系统
        self.solution = solve_sparse_cg(K, f)

        return self.solution

    def _get_boundary_nodes(self, bc_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """获取边界节点及其值"""
        tol = 1e-10
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        bc_nodes = []
        bc_values = []

        for i in range(self.n_nodes):
            x, y = self.nodes[i]
            if (abs(x - x_min) < tol or abs(x - x_max) < tol or
                    abs(y - y_min) < tol or abs(y - y_max) < tol):
                bc_nodes.append(i)
                if bc_type == 'homogeneous_dirichlet':
                    bc_values.append(0.0)
                elif bc_type == 'hot_boundary':
                    # 模拟偏滤器靶板热边界
                    if abs(y - y_min) < tol:
                        bc_values.append(100.0)  # 靶板温度 100 eV
                    else:
                        bc_values.append(10.0)  # 其他边界 10 eV
                else:
                    bc_values.append(0.0)

        return np.array(bc_nodes), np.array(bc_values)

    def compute_heat_flux(self) -> np.ndarray:
        """
        计算热流密度 q = -D . grad T

        在每个单元中心计算
        """
        q_x = np.zeros(self.n_elements)
        q_y = np.zeros(self.n_elements)

        for e in range(self.n_elements):
            elem_nodes = self.elements[e]
            coords = self.nodes[elem_nodes]
            T_elem = self.solution[elem_nodes]

            # Jacobian
            J = np.array([
                [coords[1, 0] - coords[0, 0], coords[2, 0] - coords[0, 0]],
                [coords[1, 1] - coords[0, 1], coords[2, 1] - coords[0, 1]],
            ])
            J_inv = np.linalg.inv(J)

            # 温度梯度（在参考坐标中）
            _, dN_ref = triangle_basis_values(0.0, 0.0)
            dT_dxi = dN_ref.T @ T_elem  # shape (2,)

            # 转换到物理坐标
            grad_T = J_inv.T @ dT_dxi

            # 热流: q = -D . grad T
            q = -self.D_tensor @ grad_T
            q_x[e] = q[0]
            q_y[e] = q[1]

        return np.column_stack([q_x, q_y])

    def compute_L2_error(self, exact_func: callable) -> float:
        """
        计算L2误差范数

        ||e||_2 = sqrt(integral (T_h - T_exact)^2 dA)
        """
        quad_pts, quad_wts = triangle_quadrature_7pt()

        error_sq = 0.0
        for e in range(self.n_elements):
            elem_nodes = self.elements[e]
            coords = self.nodes[elem_nodes]
            T_elem = self.solution[elem_nodes]

            J = np.array([
                [coords[1, 0] - coords[0, 0], coords[2, 0] - coords[0, 0]],
                [coords[1, 1] - coords[0, 1], coords[2, 1] - coords[0, 1]],
            ])
            det_J = abs(np.linalg.det(J))

            for q in range(len(quad_pts)):
                xi, eta = quad_pts[q]
                N, _ = triangle_basis_values(xi, eta)

                T_h = N @ T_elem
                x = coords[0, 0] + (coords[1, 0] - coords[0, 0]) * xi + (coords[2, 0] - coords[0, 0]) * eta
                y = coords[0, 1] + (coords[1, 1] - coords[0, 1]) * xi + (coords[2, 1] - coords[0, 1]) * eta
                T_exact = exact_func(x, y)

                error_sq += quad_wts[q] * det_J * (T_h - T_exact)**2

        return np.sqrt(error_sq)
