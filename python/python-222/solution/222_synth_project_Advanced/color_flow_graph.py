# -*- coding: utf-8 -*-
"""
color_flow_graph.py
===================

色流图的最小弦长配置与色连通性 Laplacian。

融合种子项目:
    - 287_dijkstra: Dijkstra 最短路径算法
    - 648_laplacian_matrix: Laplacian 矩阵及其谱性质

物理背景:
    在 parton shower 中, 每个 splitting 产生色-反色对 (q-qbar 或 g-g)。
    色流构成一个二分图 (色荷节点 ↔ 反色荷节点)。强子化时,
    弦 (string) 连接色-反色对, 总弦长决定强子化概率:
        P ~ exp(-b * L_total)
    其中 b = Lund 参数, L_total 为所有弦的总长度。

    最小化 L_total 等价于在色流图上求最小权匹配。
    对于树状色流 (典型情况), 可用 Dijkstra 算法高效求解。

    色 Laplacian L_c 定义在色流图上:
        L_{ij} = -w_{ij}  (i != j, 有连接)
        L_{ii} = sum_{j!=i} w_{ij}
    其谱 (Fiedler 值) 反映色连通性与强子化多重数。
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Set, Optional
import constants as C


# ======================================================================
# Parton 色荷节点
# ======================================================================
class ColorNode:
    """色荷节点: 夸克 (q), 反夸克 (qbar), 胶子 (g = c-cbar)"""
    __slots__ = ('id', 'ptype', 'color', 'anticolor', 'energy', 'eta', 'phi', 'px', 'py', 'pz', 'E')

    def __init__(self, id: int, ptype: str, color: int, anticolor: int = 0,
                 energy: float = 1.0, eta: float = 0.0, phi: float = 0.0,
                 px: float = 0.0, py: float = 0.0, pz: float = 0.0, E: float = 1.0):
        self.id = id
        self.ptype = ptype  # 'q', 'qbar', 'g'
        self.color = color
        self.anticolor = anticolor
        self.energy = energy
        self.eta = eta
        self.phi = phi
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E


def deltaR(node1: ColorNode, node2: ColorNode) -> float:
    """
    快度-方位角距离:
        Delta R = sqrt( (eta1 - eta2)^2 + (phi1 - phi2)^2 )
    用于衡量色弦的"长度" (在探测器几何中)。
    """
    deta = node1.eta - node2.eta
    dphi = node1.phi - node2.phi
    # 周期性: phi in [-pi, pi]
    while dphi > C.PI:
        dphi -= 2.0 * C.PI
    while dphi < -C.PI:
        dphi += 2.0 * C.PI
    return math.sqrt(deta * deta + dphi * dphi)


def invariant_mass(n1: ColorNode, n2: ColorNode) -> float:
    """
    两 parton 的不变质量:
        m^2 = (E1+E2)^2 - (p1+p2)^2
    """
    E = n1.E + n2.E
    px = n1.px + n2.px
    py = n1.py + n2.py
    pz = n1.pz + n2.pz
    m2 = E * E - px * px - py * py - pz * pz
    return math.sqrt(max(0.0, m2))


# ======================================================================
# Dijkstra 最短路径 (来自 287_dijkstra)
# ======================================================================
def dijkstra_min_distance(nv: int, ohd: List[List[int]]) -> List[int]:
    """
    Dijkstra 最短路径算法, 从节点 0 出发。
    直接移植自 287_dijkstra/dijkstra_distance.m。

    Parameters
    ----------
    nv : int
        节点数
    ohd : list of list of int
        ohd[i][j] = 从 i 到 j 的直接距离; INF 表示不连通

    Returns
    -------
    list of int
        mind[i] = 从节点 0 到节点 i 的最短距离
    """
    INF = 10**9
    connected = [0] * nv
    connected[0] = 1
    mind = [INF] * nv
    for i in range(1, nv):
        mind[i] = ohd[0][i]

    for _ in range(nv - 1):
        # 找未连接中距离最小的节点
        mv = -1
        mv_dist = INF
        for i in range(nv):
            if connected[i] == 0 and mind[i] < mv_dist:
                mv = i
                mv_dist = mind[i]

        if mv < 0:
            break
        connected[mv] = 1

        # 更新邻居
        for i in range(nv):
            if connected[i] == 0:
                new_dist = mind[mv] + ohd[mv][i]
                if new_dist < mind[i]:
                    mind[i] = new_dist

    return mind


def dijkstra_path(nv: int, ohd: List[List[float]], source: int, target: int) -> Tuple[float, List[int]]:
    """带路径重建的 Dijkstra (浮点权重)"""
    INF = float('inf')
    dist = [INF] * nv
    prev = [-1] * nv
    visited = [False] * nv
    dist[source] = 0.0

    for _ in range(nv):
        u = -1
        u_dist = INF
        for i in range(nv):
            if not visited[i] and dist[i] < u_dist:
                u = i
                u_dist = dist[i]
        if u < 0:
            break
        visited[u] = True
        if u == target:
            break

        for v in range(nv):
            if not visited[v] and ohd[u][v] < INF:
                alt = dist[u] + ohd[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u

    # 重建路径
    path = []
    cur = target
    while cur >= 0:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return dist[target], path


# ======================================================================
# 色流图构建与最小弦配置
# ======================================================================
class ColorFlowGraph:
    """色流图: 节点为 parton, 边为色弦候选"""

    def __init__(self, nodes: List[ColorNode]):
        self.nodes = nodes
        self.nv = len(nodes)
        self.edges: Dict[Tuple[int, int], float] = {}

    def build_distance_matrix(self) -> List[List[float]]:
        """
        构造 parton 间距矩阵 (deltaR 距离)。
        色-反色对之间的权重为 deltaR; 同色荷之间为 INF。
        """
        INF = float('inf')
        ohd = [[INF] * self.nv for _ in range(self.nv)]

        for i in range(self.nv):
            for j in range(i + 1, self.nv):
                ni, nj = self.nodes[i], self.nodes[j]
                # 色连接条件: 一个的 color 等于另一个的 anticolor
                can_connect = False
                if ni.ptype in ('q', 'g') and nj.ptype in ('qbar', 'g'):
                    if ni.color == nj.anticolor or ni.anticolor == nj.color:
                        can_connect = True
                if nj.ptype in ('q', 'g') and ni.ptype in ('qbar', 'g'):
                    if nj.color == ni.anticolor or nj.anticolor == ni.color:
                        can_connect = True
                # 也允许胶子自环 (g 作为 c-cbar)
                if ni.ptype == 'g' and nj.ptype == 'g':
                    if ni.anticolor == nj.color:
                        can_connect = True
                    if ni.color == nj.anticolor:
                        can_connect = True

                if can_connect:
                    d = deltaR(ni, nj)
                    ohd[i][j] = d
                    ohd[j][i] = d
                    self.edges[(i, j)] = d
                    self.edges[(j, i)] = d

        return ohd

    def minimal_string_configuration(self) -> List[Tuple[int, int]]:
        """
        贪心最小弦配置:
            按 deltaR 升序选择色弦, 每个节点最多被一条弦使用。
        这是最小权匹配的贪心近似 (对树状色流精确)。
        """
        sorted_edges = sorted(self.edges.items(), key=lambda x: x[1])
        used: Set[int] = set()
        strings = []

        for (i, j), d in sorted_edges:
            if i < j and i not in used and j not in used:
                strings.append((i, j))
                used.add(i)
                used.add(j)

        return strings

    def total_string_length(self, strings: List[Tuple[int, int]]) -> float:
        """总弦长 L = sum deltaR"""
        return sum(self.edges.get((i, j), 0.0) + self.edges.get((j, i), 0.0)
                   for i, j in strings) / 2.0


# ======================================================================
# 色 Laplacian 矩阵 (来自 648_laplacian_matrix)
# ======================================================================
def color_laplacian(graph: ColorFlowGraph) -> List[List[float]]:
    """
    色流图的加权 Laplacian 矩阵:
        L_{ij} = -w_{ij}    (i != j)
        L_{ii} = sum_{j!=i} w_{ij}

    对应 648_laplacian_matrix/l1dd.m 的图论推广。
    """
    n = graph.nv
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                w = graph.edges.get((i, j), 0.0)
                L[i][j] = -w
                L[i][i] += w
    return L


def fiedler_value(L: List[List[float]]) -> float:
    """
    Fiedler 值 (Laplacian 第二小特征值):
    反映图的代数连通性。
    在色流图中, 小 Fiedler 值意味着弱色连通 → 多弦碎裂。

    使用幂迭代求最小两个特征值。
    """
    n = len(L)
    if n < 2:
        return 0.0

    # 幂迭代求最大特征值 (L 是 PSD, 最大特征值 = 谱半径)
    import random
    random.seed(42)

    # 用移位幂迭代求最小特征值
    # L_max 估计
    v = [random.gauss(0, 1) for _ in range(n)]
    norm = math.sqrt(sum(x*x for x in v))
    v = [x / norm for x in v]

    lambda_max = 0.0
    for _ in range(100):
        w = _matvec(L, v)
        lambda_max = sum(v[i] * w[i] for i in range(n))
        norm = math.sqrt(sum(x*x for x in w))
        if norm < 1e-15:
            break
        v = [x / norm for x in w]

    # 移位矩阵: M = lambda_max * I - L, 最小特征值 -> 最大
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = -L[i][j]
        M[i][i] += lambda_max

    # 幂迭代求 M 的最大特征值
    v = [random.gauss(0, 1) for _ in range(n)]
    norm = math.sqrt(sum(x*x for x in v))
    v = [x / norm for x in v]

    sigma_max = 0.0
    for _ in range(200):
        w = _matvec(M, v)
        sigma_max = sum(v[i] * w[i] for i in range(n))
        norm = math.sqrt(sum(x*x for x in w))
        if norm < 1e-15:
            break
        v = [x / norm for x in w]

    lambda_min = lambda_max - sigma_max
    return max(0.0, lambda_min)


def _matvec(A: List[List[float]], v: List[float]) -> List[float]:
    n = len(A)
    w = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * v[j]
        w[i] = s
    return w


# ======================================================================
# 弦碎裂概率 (Lund 模型)
# ======================================================================
def string_fragmentation_probability(L_total: float, n_strings: int) -> float:
    """
    Lund 弦碎裂权重:
        W ~ exp(-b * L_total) * (a / L_total)^n_strings
    其中 a, b 为 Lund 参数。
    """
    if L_total <= 0:
        return 0.0
    log_w = -C.B_LUND * L_total + n_strings * math.log(max(C.A_LUND, 1e-10))
    if log_w < -500:
        return 0.0
    return math.exp(log_w)


# ======================================================================
# 示例色流事件
# ======================================================================
def make_qqbar_event(n_gluons: int = 3) -> ColorFlowGraph:
    """
    构造 e+e- -> q qbar + n gluons 事件的色流图。
    大 N_c 极限下, 色流为单线: q - g1 - g2 - ... - qbar。
    """
    nodes = []
    color_counter = 1

    # q: 色 1, 反色 0
    nodes.append(ColorNode(0, 'q', color=1, anticolor=0,
                           energy=10.0, eta=0.0, phi=0.0,
                           px=10.0, py=0.0, pz=0.0, E=10.0))

    # gluons: 色 i+1, 反色 i
    for g in range(n_gluons):
        c = color_counter + 1
        ac = color_counter
        color_counter += 1
        eta = -2.0 + 4.0 * (g + 1) / (n_gluons + 1)
        phi = 0.5 * g
        nodes.append(ColorNode(g + 1, 'g', color=c, anticolor=ac,
                               energy=5.0, eta=eta, phi=phi))

    # qbar: 色 0, 反色 color_counter + 1
    nodes.append(ColorNode(n_gluons + 1, 'qbar', color=0,
                           anticolor=color_counter + 1,
                           energy=10.0, eta=0.0, phi=C.PI,
                           px=-10.0, py=0.0, pz=0.0, E=10.0))

    graph = ColorFlowGraph(nodes)
    graph.build_distance_matrix()
    return graph


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== Color Flow Graph Test ===")
    graph = make_qqbar_event(n_gluons=4)
    print(f"  Nodes: {graph.nv}")

    strings = graph.minimal_string_configuration()
    L_total = graph.total_string_length(strings)
    print(f"  Minimal strings: {strings}")
    print(f"  Total string length: {L_total:.4f}")

    w_prob = string_fragmentation_probability(L_total, len(strings))
    print(f"  Fragmentation weight: {w_prob:.4e}")

    L = color_laplacian(graph)
    fv = fiedler_value(L)
    print(f"  Fiedler value: {fv:.6f}")

    # Dijkstra test
    print("\n=== Dijkstra Test ===")
    nv = 5
    ohd = [[10**9]*nv for _ in range(nv)]
    ohd[0][1] = 3; ohd[0][2] = 5
    ohd[1][2] = 2; ohd[1][3] = 6
    ohd[2][3] = 1; ohd[2][4] = 4
    ohd[3][4] = 2
    for i in range(nv):
        ohd[i][i] = 0
    mind = dijkstra_min_distance(nv, ohd)
    print(f"  Min distances from 0: {mind}")
