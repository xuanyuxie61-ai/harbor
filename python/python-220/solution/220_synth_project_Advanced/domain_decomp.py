"""
domain_decomp.py — 区域分解与子域管理
============================================
来源项目: 874_ply_to_tri_surface (网格分区与扇形三角化)

本模块实现:
  1. 结构化网格的区域分解 (递归坐标二分)
  2. 子域提取与接口检测
  3. 子域图结构 (邻接关系)
  4. 全局-局部节点映射

区域分解策略:
  对 N = p² 个子域, 使用递归坐标二分 (Recursive Coordinate Bisection):
    1. 按 x 坐标二分 → 左/右两半
    2. 每半按 y 坐标二分 → 四个象限
    3. 递归至 N 个子域

  等价于将网格均匀切分为 p × p 的子网格.

接口类型:
  - 面接口 (face): 两个子域共享的边
  - 边接口 (edge): 多个子域共享的点
  - 角接口 (corner): 四个子域共享的角点

ADMM 中的角色:
  每个子域是 ADMM 的一个局部问题.
  接口上的共识变量 z 保证相邻子域解的一致性.
  增广 Lagrangian 通过罚项 ρ/2 ||x_i - z||² 实现.
"""

import numpy as np
from typing import Tuple, List, Dict, Set, Optional
from config import EPS_NUM
from mesh import TriMesh6


# ============================================================
#  子域数据结构
# ============================================================
class Subdomain:
    """子域数据结构

    Attributes:
        subdomain_id: 子域编号
        global_node_ids: 子域包含的全局节点编号
        local_to_global: 局部→全局节点映射
        global_to_local: 全局→局部节点映射
        interface_nodes: 接口节点 (全局编号)
        interior_nodes: 内部节点 (全局编号)
        neighbors: 相邻子域编号
        n_local_nodes: 局部节点数
        n_interface_nodes: 接口节点数
    """

    def __init__(self, subdomain_id: int):
        self.subdomain_id = subdomain_id
        self.global_node_ids: List[int] = []
        self.local_to_global: Dict[int, int] = {}
        self.global_to_local: Dict[int, int] = {}
        self.interface_nodes: Set[int] = set()
        self.interior_nodes: Set[int] = set()
        self.neighbors: Set[int] = set()
        self.n_local_nodes = 0
        self.n_interface_nodes = 0

    def build_mappings(self):
        """构建局部-全局映射"""
        self.n_local_nodes = len(self.global_node_ids)
        self.n_interface_nodes = len(self.interface_nodes)
        self.interior_nodes = set(self.global_node_ids) - self.interface_nodes

        for local_id, global_id in enumerate(self.global_node_ids):
            self.local_to_global[local_id] = global_id
            self.global_to_local[global_id] = local_id

    def get_local_nodes(self) -> np.ndarray:
        """获取局部节点编号数组"""
        return np.arange(self.n_local_nodes)

    def get_interface_local_ids(self) -> List[int]:
        """获取接口节点的局部编号"""
        return [self.global_to_local[g] for g in self.interface_nodes
                if g in self.global_to_local]


# ============================================================
#  区域分解
# ============================================================
def decompose_rectangular_mesh(nx: int, ny: int, n_sub_x: int, n_sub_y: int) -> List[Subdomain]:
    """将 nx × ny 矩形网格分解为 n_sub_x × n_sub_y 个子域

    递归坐标二分:
      将 [0, nx] × [0, ny] 均匀切分为 n_sub_x × n_sub_y 块.
      每块包含 (nx/n_sub_x + 1) × (ny/n_sub_y + 1) 个节点
      (接口节点被相邻子域共享).

    Args:
        nx, ny: 每方向节点数减 1 (单元数)
        n_sub_x, n_sub_y: 每方向子域数

    Returns:
        List[Subdomain]: 子域列表
    """
    if nx % n_sub_x != 0 or ny % n_sub_y != 0:
        raise ValueError(f"网格 ({nx}x{ny}) 不能被子域划分 ({n_sub_x}x{n_sub_y}) 整除")

    block_nx = nx // n_sub_x  # 每子域的单元数
    block_ny = ny // n_sub_y

    n_subdomains = n_sub_x * n_sub_y
    subdomains = []

    for sy in range(n_sub_y):
        for sx in range(n_sub_x):
            sid = sy * n_sub_x + sx
            sub = Subdomain(sid)

            # 子域包含的节点范围
            x_start = sx * block_nx
            x_end = (sx + 1) * block_nx
            y_start = sy * block_ny
            y_end = (sy + 1) * block_ny

            # 收集子域节点 (含接口)
            for j in range(y_start, y_end + 1):
                for i in range(x_start, x_end + 1):
                    global_id = j * (nx + 1) + i
                    sub.global_node_ids.append(global_id)

            # 检测接口节点
            for j in range(y_start, y_end + 1):
                for i in range(x_start, x_end + 1):
                    global_id = j * (nx + 1) + i
                    is_interface = False
                    if i == x_start and sx > 0:
                        is_interface = True
                    if i == x_end and sx < n_sub_x - 1:
                        is_interface = True
                    if j == y_start and sy > 0:
                        is_interface = True
                    if j == y_end and sy < n_sub_y - 1:
                        is_interface = True
                    if is_interface:
                        sub.interface_nodes.add(global_id)

            sub.build_mappings()

            # 邻居
            if sx > 0:
                sub.neighbors.add(sy * n_sub_x + (sx - 1))
            if sx < n_sub_x - 1:
                sub.neighbors.add(sy * n_sub_x + (sx + 1))
            if sy > 0:
                sub.neighbors.add((sy - 1) * n_sub_x + sx)
            if sy < n_sub_y - 1:
                sub.neighbors.add((sy + 1) * n_sub_x + sx)

            subdomains.append(sub)

    return subdomains


# ============================================================
#  接口管理
# ============================================================
class InterfaceManager:
    """管理子域间接口

    ADMM 中的角色:
      - 每个接口关联一组共识变量 z
      - 局部变量 x_i 在接口处必须等于 z
      - 对偶变量 λ 跟踪接口一致性违反

    接口图结构:
      interface_edges: [(sub_i, sub_j, shared_nodes)]
      interface_nodes_global: 所有接口节点的全局集合
    """

    def __init__(self, subdomains: List[Subdomain]):
        self.subdomains = subdomains
        self.interface_edges: List[Tuple[int, int, Set[int]]] = []
        self.node_to_interfaces: Dict[int, List[Tuple[int, int]]] = {}
        self._build_interfaces()

    def _build_interfaces(self):
        """构建接口列表"""
        n = len(self.subdomains)
        seen = set()

        for i in range(n):
            for j in range(i + 1, n):
                shared = self.subdomains[i].interface_nodes & self.subdomains[j].interface_nodes
                if shared:
                    self.interface_edges.append((i, j, shared))
                    for node in shared:
                        if node not in self.node_to_interfaces:
                            self.node_to_interfaces[node] = []
                        self.node_to_interfaces[node].append((i, j))
                    seen.update(shared)

    @property
    def n_interfaces(self) -> int:
        return len(self.interface_edges)

    @property
    def all_interface_nodes(self) -> Set[int]:
        nodes = set()
        for _, _, shared in self.interface_edges:
            nodes.update(shared)
        return nodes

    def get_interface_nodes_for_subdomain(self, sub_id: int) -> Set[int]:
        """获取某子域的所有接口节点"""
        return self.subdomains[sub_id].interface_nodes

    def get_adjacent_subdomains(self, sub_id: int) -> Set[int]:
        """获取某子域的相邻子域"""
        return self.subdomains[sub_id].neighbors


# ============================================================
#  全局-局部映射
# ============================================================
def build_global_assembly_map(subdomains: List[Subdomain],
                               n_global_nodes: int) -> Dict[int, List[Tuple[int, int]]]:
    """构建全局组装映射

    对每个全局节点, 记录其在哪些子域中的哪个局部位置.
    用于 ADMM 中的共识约束组装.

    Returns:
        Dict: global_node → [(sub_id, local_id), ...]
    """
    gmap: Dict[int, List[Tuple[int, int]]] = {}
    for sub in subdomains:
        for global_id, local_id in sub.global_to_local.items():
            if global_id not in gmap:
                gmap[global_id] = []
            gmap[global_id].append((sub.subdomain_id, local_id))
    return gmap


# ============================================================
#  子域网格提取
# ============================================================
def extract_subdomain_mesh(full_mesh: TriMesh6, subdomain: Subdomain) -> dict:
    """从全局网格提取子域局部网格

    Returns:
        dict: nodes, elements, boundary_nodes
    """
    local_nodes = []
    for gid in subdomain.global_node_ids:
        if gid < full_mesh.n_nodes:
            local_nodes.append(full_mesh.nodes[gid])
        else:
            local_nodes.append(np.zeros(2))  # 占位

    local_nodes = np.array(local_nodes) if local_nodes else np.zeros((0, 2))

    # 提取完全属于子域的单元
    local_elements = []
    global_set = set(subdomain.global_node_ids)
    for e in range(full_mesh.n_elements):
        elem_nodes = set(full_mesh.elements[e])
        if elem_nodes.issubset(global_set):
            # 转换为局部编号
            local_elem = [subdomain.global_to_local.get(n, -1) for n in full_mesh.elements[e]]
            if all(n >= 0 for n in local_elem):
                local_elements.append(local_elem)

    local_elements = np.array(local_elements) if local_elements else np.zeros((0, 6), dtype=np.int64)

    return {
        'nodes': local_nodes,
        'elements': local_elements,
        'n_nodes': len(local_nodes),
        'n_elements': len(local_elements),
    }


# ============================================================
#  2D 规则网格上的简化分解 (用于 ADMM 演示)
# ============================================================
def decompose_grid_2d(nx: int, ny: int, n_subdomains: int) -> dict:
    """2D 网格的简化分解 (用于 ADMM 演示)

    将 nx × ny 的 2D 网格分解为 n_subdomains 个子域.
    n_subdomains 必须为完全平方数.

    Returns:
        dict: subdomain_list, interface_map, global_shape
    """
    p = int(np.sqrt(n_subdomains))
    if p * p != n_subdomains:
        raise ValueError(f"子域数必须为完全平方数, got {n_subdomains}")

    if nx % p != 0 or ny % p != 0:
        raise ValueError(f"网格 ({nx}x{ny}) 不能被子域数 {n_subdomains} 整除")

    block_nx = nx // p
    block_ny = ny // p

    subdomain_list = []
    interface_map: Dict[int, List[Tuple[int, int]]] = {}

    for sy in range(p):
        for sx in range(p):
            sid = sy * p + sx
            # 子域节点范围
            i_start = sx * block_nx
            i_end = (sx + 1) * block_nx
            j_start = sy * block_ny
            j_end = (sy + 1) * block_ny

            nodes = []
            for j in range(j_start, i_end if j_end > i_end else j_end + 1):
                for i in range(i_start, i_end + 1):
                    nodes.append(j * (nx + 1) + i)

            subdomain_list.append({
                'id': sid,
                'i_range': (i_start, i_end),
                'j_range': (j_start, j_end),
                'nodes': nodes,
                'n_local': len(nodes),
            })

    return {
        'subdomain_list': subdomain_list,
        'p': p,
        'global_shape': (nx + 1, ny + 1),
        'block_shape': (block_nx, block_ny),
    }
