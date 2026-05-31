"""
molecular_topology.py
分子拓扑与图结构分析模块

核心功能：
- XYZ 分子坐标文件解析
- 基于距离截断的键合图构建
- 邻接表与 METIS 图格式输出
- 分子连通性分析

科学背景：
酶是大型生物分子（通常 10³-10⁵ 原子），其拓扑结构决定了：
    - 活性位点的几何构型
    - 底物结合口袋的形状
    - 催化残基的空间排布

分子图 G = (V, E)：
    V: 原子集合 {1, 2, ..., N}
    E: 化学键集合 {(i,j) | |r_i - r_j| < r_cut(i,j)}

其中截断距离通常取：
    r_cut = 1.2 * (R_i + R_j)
    R_i, R_j 为原子共价半径

METIS 图格式：
    第一行: <顶点数> <边数>
    第 i+1 行: 顶点 i 的邻接顶点列表

在过渡态搜索中，分子拓扑用于：
    1. 定义柔性坐标（二面角、键角）
    2. 构建反应坐标的原子子集
    3. 识别催化三联体（如 Ser-His-Asp）
    4. 图划分用于并行计算
"""

import numpy as np


# 原子共价半径（Å，CCB 数据）
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
    'P': 1.07, 'F': 0.57, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'Fe': 1.32, 'Zn': 1.22, 'Mg': 1.41, 'Ca': 1.76, 'Na': 1.66,
    'K': 1.96, 'Cu': 1.32, 'Mn': 1.39, 'Co': 1.26, 'Ni': 1.21
}

# 默认半径（未知元素）
DEFAULT_RADIUS = 1.5


class XYZParser:
    """XYZ 格式分子坐标解析器"""

    @staticmethod
    def parse_string(xyz_string):
        """
        从字符串解析 XYZ 数据

        XYZ 格式：
            <原子数>
            <注释行>
            <元素> <x> <y> <z>
            ...
        """
        lines = xyz_string.strip().split('\n')
        if len(lines) < 3:
            raise ValueError("XYZ 数据格式错误")

        n_atoms = int(lines[0].strip())
        # lines[1] 为注释行
        atoms = []
        coords = []

        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            if len(parts) < 4:
                raise ValueError(f"XYZ 第 {i+1} 行格式错误")
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(element)
            coords.append([x, y, z])

        return atoms, np.array(coords, dtype=float)

    @staticmethod
    def generate_demo_xyz():
        """生成演示用酶活性位点 XYZ 数据（丙酮酸脱氢酶 E1 组分简化模型）"""
        # 简化活性位点：底物类似物 + 催化残基
        xyz_data = """18
demo enzyme active site
C   0.000   0.000   0.000
O   1.200   0.000   0.000
O  -0.800   0.900   0.000
C  -0.500  -1.200   0.000
N   1.500   1.200   0.500
C   2.800   1.300   0.200
O   3.500   0.400  -0.100
C   3.200   2.600   0.300
N  -1.500   0.200   0.800
C  -2.200   1.300   1.200
O  -1.800   2.400   1.500
C  -3.600   1.100   1.200
O   0.200  -2.300   0.200
H  -1.100  -1.200  -0.900
H   0.100   0.800   0.800
H   3.700   3.100  -0.500
H   2.600   3.200   1.000
S  -4.500   2.400   0.800
"""
        return xyz_data


class MolecularGraph:
    """分子图结构"""

    def __init__(self, atoms, coordinates):
        self.atoms = atoms
        self.coords = np.asarray(coordinates, dtype=float)
        self.n_atoms = len(atoms)
        self.adjacency = [[] for _ in range(self.n_atoms)]
        self.bonds = []

    def build_bonds(self, scale_factor=1.2):
        """
        基于共价半径构建化学键

        键合判据：
            |r_i - r_j| < scale_factor * (R_i + R_j)
        """
        for i in range(self.n_atoms):
            ri = COVALENT_RADII.get(self.atoms[i], DEFAULT_RADIUS)
            for j in range(i + 1, self.n_atoms):
                rj = COVALENT_RADII.get(self.atoms[j], DEFAULT_RADIUS)
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                cutoff = scale_factor * (ri + rj)
                if dist < cutoff and dist > 0.3:
                    self.adjacency[i].append(j)
                    self.adjacency[j].append(i)
                    self.bonds.append((i, j, dist))

    def get_degree(self, i):
        """顶点度"""
        return len(self.adjacency[i])

    def get_neighbors(self, i):
        """邻接顶点"""
        return self.adjacency[i].copy()

    def metis_format(self):
        """
        生成 METIS 图格式字符串

        格式：
            N M
            v1 v2 ... vk
            ...
        其中 M 为无向边数（每条边只计一次）
        """
        n_edges = len(self.bonds)
        lines = [f"{self.n_atoms} {n_edges}"]
        for i in range(self.n_atoms):
            neighbors = sorted(self.adjacency[i])
            if neighbors:
                lines.append(" ".join(str(n + 1) for n in neighbors))  # METIS 使用 1-based 索引
            else:
                lines.append("")
        return "\n".join(lines)

    def connected_components(self):
        """查找连通分量（用于检查分子是否断裂）"""
        visited = [False] * self.n_atoms
        components = []

        for start in range(self.n_atoms):
            if not visited[start]:
                comp = []
                stack = [start]
                visited[start] = True
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    for neigh in self.adjacency[node]:
                        if not visited[neigh]:
                            visited[neigh] = True
                            stack.append(neigh)
                components.append(comp)

        return components

    def find_catalytic_triad(self):
        """
        搜索催化三联体（Ser-His-Asp/Glu）模式

        典型模式：
            Asp/Glu (羧基) -- His (咪唑) -- Ser (羟基)
            几何要求：Asp-His ~ 2.8 Å, His-Ser ~ 2.8 Å
        """
        triads = []
        for i in range(self.n_atoms):
            if self.atoms[i] in ['O', 'N']:
                for j in self.adjacency[i]:
                    if self.atoms[j] in ['C', 'N']:
                        for k in self.adjacency[j]:
                            if k != i and self.atoms[k] in ['O', 'N', 'S']:
                                d_ik = np.linalg.norm(self.coords[i] - self.coords[k])
                                if 2.5 < d_ik < 4.0:
                                    triads.append((i, j, k, d_ik))
        return triads

    def subgraph_around_atom(self, center_idx, radius=3):
        """
        提取中心原子周围 radius 跳内的子图
        用于定义局部反应坐标
        """
        visited = {center_idx}
        frontier = {center_idx}
        for _ in range(radius):
            new_frontier = set()
            for node in frontier:
                for neigh in self.adjacency[node]:
                    if neigh not in visited:
                        visited.add(neigh)
                        new_frontier.add(neigh)
            frontier = new_frontier
        return sorted(list(visited))


def analyze_molecular_topology(atoms, coords):
    """
    完整分子拓扑分析管道
    """
    graph = MolecularGraph(atoms, coords)
    graph.build_bonds()

    results = {
        'n_atoms': graph.n_atoms,
        'n_bonds': len(graph.bonds),
        'components': graph.connected_components(),
        'n_components': len(graph.connected_components()),
        'triads': graph.find_catalytic_triad(),
        'metis_graph': graph.metis_format()
    }

    # 计算平均配位数
    avg_degree = np.mean([graph.get_degree(i) for i in range(graph.n_atoms)])
    results['avg_degree'] = avg_degree

    return graph, results
