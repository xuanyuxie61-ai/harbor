"""
mesh_generator.py
=================
径向网格生成与稀疏矩阵重排序模块。

融合原项目 1016_rcm（Reverse Cuthill-McKee 带宽缩减算法）与
393_fem1d_lagrange（一维 Lagrange 有限元刚度矩阵组装）的核心思想，
为球坐标一维拉格朗日流体动力学生成计算网格，并构建带质量 lumping
的有限元离散算子。

在ICF内爆中，网格需满足以下约束：
1. 烧蚀前缘附近网格加密（高温度/密度梯度）
2. 拉格朗日坐标下网格随物质运动
3. 有限元刚度矩阵需保持稀疏、对称、正定
"""

import numpy as np
from typing import Tuple, List
from utils import spherical_volume, safe_divide
from icf_parameters import TP, NP


class RadialMesh:
    """
    一维球对称径向网格。
    节点坐标 r[0..n] 定义 n 个单元（球壳）。
    """

    def __init__(self, n_cells: int = NP.N_RADIAL):
        self.n_cells = n_cells
        self.n_nodes = n_cells + 1
        self.r = np.zeros(self.n_nodes)
        self._generate_initial_mesh()

    def _generate_initial_mesh(self):
        """
        生成初始非均匀网格：
        - 烧蚀层外缘附近加密（R_ABLATION 处）
        - DT冰层内部适中分辨率
        - 充气腔内部较疏
        """
        n = self.n_cells
        # 使用复合映射：在烧蚀前缘附近增加分辨率
        # s in [0, 1] 为参数坐标
        s = np.linspace(0.0, 1.0, self.n_nodes)

        # 映射函数：在 s ≈ 0.9（对应烧蚀层外缘）附近加密
        # 采用 sigmoid-like 映射
        alpha = 8.0
        beta = 0.85
        s_mapped = s + alpha * s * (1.0 - s) * (s - beta) / (1.0 + alpha * 0.25)
        s_mapped = np.clip(s_mapped, 0.0, 1.0)
        s_mapped = (s_mapped - s_mapped[0]) / (s_mapped[-1] - s_mapped[0])

        self.r = s_mapped * TP.R_ABLATION

        # 保证关键界面处存在节点
        self._enforce_interface_nodes()

    def _enforce_interface_nodes(self):
        """确保 R_DT_ICE 和 R_GAS 界面处有节点。"""
        for target in [TP.R_DT_ICE, TP.R_GAS]:
            idx = np.searchsorted(self.r, target)
            if idx > 0 and idx < self.n_nodes:
                if not np.isclose(self.r[idx], target):
                    # 插入节点
                    self.r = np.insert(self.r, idx, target)
                    self.n_nodes += 1
                    self.n_cells += 1

    def cell_volumes(self) -> np.ndarray:
        """计算每个球壳单元的体积。"""
        vol = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            vol[i] = spherical_volume(self.r[i], self.r[i + 1])
        return vol

    def cell_centers(self) -> np.ndarray:
        """计算单元中心位置（体积加权）。"""
        centers = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            r1, r2 = self.r[i], self.r[i + 1]
            # 球壳体积加权中心: 3/4 * (r2^4 - r1^4) / (r2^3 - r1^3)
            num = r2**4 - r1**4
            den = r2**3 - r1**3
            centers[i] = 0.75 * safe_divide(np.array([num]), np.array([den]))[0]
            if den < 1.0e-30:
                centers[i] = 0.5 * (r1 + r2)
        return centers

    def cell_widths(self) -> np.ndarray:
        """计算单元径向宽度 dr。"""
        return np.diff(self.r)

    def get_material_zone(self, cell_idx: int) -> str:
        """判断单元所属材料区。"""
        rc = self.cell_centers()[cell_idx]
        if rc >= TP.R_DT_ICE:
            return "ablator"
        elif rc >= TP.R_GAS:
            return "dt_ice"
        else:
            return "gas"

    def get_density_by_zone(self, cell_idx: int) -> float:
        """根据材料区返回参考密度。"""
        zone = self.get_material_zone(cell_idx)
        return {"ablator": TP.RHO_CH, "dt_ice": TP.RHO_DT, "gas": TP.RHO_GAS}[zone]

    def remap_lagrangian(self, mass: np.ndarray):
        """
        根据质量守恒重新映射拉格朗日坐标。
        r_new[i+1]^3 = r_new[i]^3 + 3 * m[i] / (4 * pi * rho)
        这里采用等密度近似。
        """
        new_r = np.zeros_like(self.r)
        new_r[0] = self.r[0]
        for i in range(self.n_cells):
            rho = self.get_density_by_zone(i)
            vol = mass[i] / rho
            new_r[i + 1] = (new_r[i]**3 + 3.0 * vol / (4.0 * np.pi))**(1.0 / 3.0)
        self.r = new_r


# ========================================================================
# RCM 稀疏矩阵重排序（基于原项目 1016_rcm）
# ========================================================================

def rcm_ordering(adjacency: List[List[int]], n: int) -> np.ndarray:
    """
    Reverse Cuthill-McKee (RCM) 算法实现。
    对一维有限元刚度图进行节点重排序，以减少矩阵带宽。

    参数
    ----
    adjacency : List[List[int]]
        邻接表，adjacency[i] 为节点 i 的邻居列表
    n : int
        节点数

    返回
    ----
    perm : np.ndarray
        重排序后的节点编号（新 -> 旧）
    """
    if n <= 0:
        return np.array([], dtype=int)

    # 构建 mask（未编号=1，已编号=0）
    mask = np.ones(n, dtype=int)
    perm = np.zeros(n, dtype=int)
    num = 0

    for start in range(n):
        if mask[start] == 0:
            continue

        # 找一个伪边缘节点作为根
        root = _pseudo_peripheral_node(adjacency, n, mask, start)

        # 层次遍历（BFS）生成 Cuthill-McKee 序
        level_order = _bfs_level_order(adjacency, n, mask, root)

        # 反转得到 RCM 序
        level_order.reverse()

        for node in level_order:
            perm[num] = node
            mask[node] = 0
            num += 1

        if num >= n:
            break

    return perm


def _pseudo_peripheral_node(adjacency, n, mask, start):
    """寻找伪边缘节点：迭代BFS直到层数不再增加。"""
    root = start
    while True:
        level_nodes, level_ptr = _build_level_structure(adjacency, n, mask, root)
        # 取最后一层中度数最小的节点
        last_level = level_nodes[level_ptr[-2]:level_ptr[-1]] if len(level_ptr) > 1 else [root]
        if not last_level:
            break
        min_deg_node = min(last_level, key=lambda x: len(adjacency[x]))
        if min_deg_node == root:
            break
        root = min_deg_node
    return root


def _build_level_structure(adjacency, n, mask, root):
    """构建层次结构，返回层内节点列表和层指针。"""
    visited = np.zeros(n, dtype=int)
    level_nodes = []
    level_ptr = [0]
    queue = [root]
    visited[root] = 1

    while queue:
        level_ptr.append(len(level_nodes) + len(queue))
        next_queue = []
        for node in queue:
            level_nodes.append(node)
            for nb in adjacency[node]:
                if visited[nb] == 0 and mask[nb] == 1:
                    visited[nb] = 1
                    next_queue.append(nb)
        queue = next_queue

    return np.array(level_nodes, dtype=int), level_ptr


def _bfs_level_order(adjacency, n, mask, root):
    """BFS生成层次遍历序（按度排序每层）。"""
    visited = np.zeros(n, dtype=int)
    order = []
    queue = [root]
    visited[root] = 1
    mask[root] = 0

    while queue:
        # 按度排序当前层
        queue.sort(key=lambda x: len(adjacency[x]))
        next_queue = []
        for node in queue:
            order.append(node)
            for nb in adjacency[node]:
                if visited[nb] == 0 and mask[nb] == 1:
                    visited[nb] = 1
                    mask[nb] = 0
                    next_queue.append(nb)
        queue = next_queue

    return order


def build_1d_fem_adjacency(n_nodes: int) -> List[List[int]]:
    """构建一维有限元（线性基）的邻接表。"""
    adj = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        if i > 0:
            adj[i].append(i - 1)
        if i < n_nodes - 1:
            adj[i].append(i + 1)
    return adj


# ========================================================================
# 有限元刚度与质量矩阵（基于原项目 393_fem1d_lagrange）
# ========================================================================

def fem_stiffness_mass_1d_spherical(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    在一维球坐标下组装 Lagrange 线性基函数的刚度矩阵 A 与质量矩阵 M。

    单元 e = [r_i, r_{i+1}]，长度 h_e = r_{i+1} - r_i。
    局部刚度矩阵（球坐标拉普拉斯项）:
        K_e[i,j] = integral_{r_i}^{r_{i+1}} 4*pi*r^2 * dN_i/dr * dN_j/dr dr
    局部质量矩阵:
        M_e[i,j] = integral_{r_i}^{r_{i+1}} 4*pi*r^2 * N_i * N_j dr

    采用解析积分（线性形函数）。

    参数
    ----
    r : np.ndarray
        节点坐标

    返回
    ----
    K, M : 全局刚度矩阵与质量矩阵（三对角稀疏格式）
    """
    n = len(r) - 1  # 单元数
    n_nodes = len(r)

    # 三对角存储
    K_main = np.zeros(n_nodes)
    K_upper = np.zeros(n_nodes - 1)
    K_lower = np.zeros(n_nodes - 1)

    M_main = np.zeros(n_nodes)
    M_upper = np.zeros(n_nodes - 1)
    M_lower = np.zeros(n_nodes - 1)

    for e in range(n):
        r1, r2 = r[e], r[e + 1]
        h = r2 - r1
        if h <= 1.0e-30:
            continue

        # 球坐标体积因子 4*pi*r^2 的单元积分
        # 对线性形函数 N1=(r2-r)/h, N2=(r-r1)/h
        # int r^2 dr = h/3 * (r1^2 + r1*r2 + r2^2)
        vol_factor = (4.0 * np.pi * h / 3.0) * (r1**2 + r1 * r2 + r2**2)

        # 刚度局部矩阵: (4*pi*r^2) * (dN/dr)^2
        # 简化: 取单元中心 r_c = (r1+r2)/2 的近似
        r_c = 0.5 * (r1 + r2)
        k_local = (4.0 * np.pi * r_c**2) / h * np.array([[1.0, -1.0], [-1.0, 1.0]])

        # 质量局部矩阵 (lumped approximation + consistent)
        m_local = vol_factor / h * np.array([[1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0]])

        # 组装
        idx = [e, e + 1]
        for i_local in range(2):
            gi = idx[i_local]
            K_main[gi] += k_local[i_local, i_local]
            M_main[gi] += m_local[i_local, i_local]
            if i_local == 0:
                K_upper[e] += k_local[0, 1]
                K_lower[e] += k_local[1, 0]
                M_upper[e] += m_local[0, 1]
                M_lower[e] += m_local[1, 0]

    # 构造 CSR-like 三对角对象
    class Tridiag:
        def __init__(self, lower, main, upper):
            self.lower = lower
            self.main = main
            self.upper = upper
            self.n = len(main)

    return Tridiag(K_lower, K_main, K_upper), Tridiag(M_lower, M_main, M_upper)


def apply_rcm_to_tridiag(K, M, perm: np.ndarray):
    """
    对三对角矩阵应用 RCM 重排序。
    由于一维线性有限元本身已是带宽最优（带宽=1），
    此函数主要作为验证与教学用途。
    """
    n = K.n
    K_new_main = np.zeros(n)
    K_new_upper = np.zeros(n - 1)
    K_new_lower = np.zeros(n - 1)
    M_new_main = np.zeros(n)
    M_new_upper = np.zeros(n - 1)
    M_new_lower = np.zeros(n - 1)

    inv_perm = np.zeros(n, dtype=int)
    inv_perm[perm] = np.arange(n)

    for i in range(n):
        ii = perm[i]
        K_new_main[i] = K.main[ii]
        M_new_main[i] = M.main[ii]
        if i < n - 1:
            # 保持相邻关系
            j = i + 1
            jj = perm[j]
            if abs(ii - jj) == 1:
                idx = min(ii, jj)
                K_new_upper[i] = K.upper[idx] if ii < jj else K.lower[idx]
                K_new_lower[i] = K.lower[idx] if ii < jj else K.upper[idx]
                M_new_upper[i] = M.upper[idx] if ii < jj else M.lower[idx]
                M_new_lower[i] = M.lower[idx] if ii < jj else M.upper[idx]

    class Tridiag:
        def __init__(self, lower, main, upper):
            self.lower = lower
            self.main = main
            self.upper = upper
            self.n = len(main)

    return Tridiag(K_new_lower, K_new_main, K_new_upper), Tridiag(M_new_lower, M_new_main, M_new_upper)
