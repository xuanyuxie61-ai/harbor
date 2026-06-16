"""
channel_coupling.py
===================================================================
多道耦合反应通道图与角动量分解模块

映射种子项目:
  - 286_digraph_arc: 有向图弧表 → 反应通道耦合有向图
  - 905_pram: 组合拼图 → 角动量耦合通道组合分解

核心物理公式:
  耦合道方程 (CC):
    [d^2/dr^2 + k_i^2 - l_i(l_i+1)/r^2 - 2m/hbar^2*V_ii(r)] u_i(r)
      = sum_{j!=i} 2m/hbar^2 * V_ij(r) * u_j(r)

  耦合势矩阵 (变形核):
    V_ij(r) = sum_lambda V_lambda(r) * <i|Y_lambda|j>

  通道量子数:
    alpha = {A_target, J_target, A_proj, j_proj, l, J, pi}

  有向图表示:
    节点 = 反应通道
    弧(i,j) = 通道 i 到 j 的耦合强度 V_ij > 0

  欧拉回路条件:
    入度 = 出度 (对所有节点) => 存在闭合耦合链
===================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from itertools import combinations, product


# ---------- 物理常数 ----------
HBAR_C = 197.3269804


class ReactionChannel:
    """
    单个反应通道定义

    通道量子数:
        alpha: 通道标识符
        A_target: 靶核质量数
        J_target: 靶核自旋
        excitation: 靶核激发能 [MeV]
        l_quantum: 轨道角动量
        J_total: 总角动量 (守恒)
        parity: 宇称 (+1 或 -1)
        threshold: 道阈能 [MeV]
        channel_momentum: 道内波数 k_i [1/fm]
    """

    def __init__(
        self,
        alpha: str,
        A_target: int = 208,
        J_target: float = 0.0,
        excitation: float = 0.0,
        l_quantum: int = 0,
        J_total: float = 0.5,
        parity: int = 1,
        mass_reduced: float = 469.0,
        energy_cm: float = 14.0,
    ):
        self.alpha = alpha
        self.A_target = A_target
        self.J_target = J_target
        self.excitation = excitation
        self.l_quantum = l_quantum
        self.J_total = J_total
        self.parity = parity
        self.mass_reduced = mass_reduced
        self.energy_cm = energy_cm

        # 阈值能量
        self.threshold = excitation

        # 道内波数
        E_available = energy_cm - excitation
        if E_available > 0:
            self.k_channel = np.sqrt(2.0 * mass_reduced * E_available) / HBAR_C
            self.is_open = True
        else:
            self.k_channel = 0.0
            self.is_open = False

    def __repr__(self):
        status = "open" if self.is_open else "closed"
        return (f"Channel({self.alpha}: l={self.l_quantum}, "
                f"J={self.J_total}, E*={self.excitation}MeV, [{status}])")


class CouplingGraph:
    """
    反应通道耦合有向图
    (映射自 286_digraph_arc — 有向弧表/邻接矩阵/正星形表示)

    支持三种表示:
    1. 弧表 (edge list): [(i,j,V_ij), ...]
    2. 邻接矩阵: A[i,j] = V_ij
    3. 正星形 (forward star): 压缩稀疏行格式

    图论性质:
    - 欧拉回路: 耦合链是否存在闭合循环
    - 连通分量: 独立耦合子系统
    - 度分布: 各通道耦合度
    """

    def __init__(self):
        self.channels: List[ReactionChannel] = []
        self._arc_list: List[Tuple[int, int, complex]] = []
        self._adj_matrix: Optional[np.ndarray] = None
        self._channel_names: Dict[str, int] = {}

    def add_channel(self, channel: ReactionChannel) -> int:
        """添加反应通道, 返回通道索引"""
        idx = len(self.channels)
        self.channels.append(channel)
        self._channel_names[channel.alpha] = idx
        self._adj_matrix = None  # 重建缓存
        return idx

    def add_coupling(
        self, i: int, j: int, V_ij: complex
    ) -> None:
        """
        添加通道 i -> j 的耦合

        参数:
            i, j: 通道索引
            V_ij: 耦合势强度 (复数)
        """
        if i >= len(self.channels) or j >= len(self.channels):
            raise IndexError(f"通道索引越界: ({i}, {j}) >= {len(self.channels)}")
        self._arc_list.append((i, j, V_ij))
        self._adj_matrix = None

    def build_adjacency_matrix(self) -> np.ndarray:
        """
        构建耦合邻接矩阵 A:
            A[i,j] = V_ij  (通道 i 到 j 的耦合强度)
            A[i,i] = 0     (不自耦合)
        """
        n = len(self.channels)
        A = np.zeros((n, n), dtype=complex)
        for i, j, v in self._arc_list:
            A[i, j] = v
        self._adj_matrix = A
        return A

    def build_forward_star(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建正星形表示 (Forward Star)
        (映射自 digraph_arc_to_star)

        返回:
            arcfir: 节点 i 的第一条弧在 fwdarc 中的位置 [n+1]
            arcend: 每条弧的终点节点 [m]
            arcval: 每条弧的耦合值 [m]
        """
        n = len(self.channels)
        arcs = sorted(self._arc_list, key=lambda x: (x[0], x[1]))

        arcfir = np.zeros(n + 1, dtype=int)
        arcend = np.zeros(len(arcs), dtype=int)
        arcval = np.zeros(len(arcs), dtype=complex)

        for idx, (i, j, v) in enumerate(arcs):
            arcend[idx] = j
            arcval[idx] = v
            arcfir[i + 1] += 1

        # 前缀和
        for i in range(1, n + 1):
            arcfir[i] += arcfir[i - 1]

        return arcfir, arcend, arcval

    def compute_degrees(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算各通道的入度和出度
        (映射自 digraph_arc_degree)

        入度(i) = sum_j A[j,i]  (有多少通道耦合到 i)
        出度(i) = sum_j A[i,j]  (i 耦合到多少通道)

        返回:
            (in_degree, out_degree)
        """
        A = self.build_adjacency_matrix()
        in_degree = np.sum(np.abs(A) > 1e-15, axis=0)  # 列和
        out_degree = np.sum(np.abs(A) > 1e-15, axis=1)  # 行和
        return in_degree, out_degree

    def is_eulerian(self) -> int:
        """
        检测耦合图是否存在欧拉回路
        (映射自 digraph_arc_is_eulerian)

        返回:
            0: 非欧拉
            1: 存在开欧拉迹
            2: 存在闭欧拉回路

        条件:
            闭欧拉回路: 对所有节点, 入度 = 出度
            开欧拉迹: 恰好一个节点入度=出度+1, 一个节点出度=入度+1
        """
        in_deg, out_deg = self.compute_degrees()
        diff = in_deg - out_deg

        n_zero = np.sum(diff == 0)
        n_plus = np.sum(diff == 1)
        n_minus = np.sum(diff == -1)

        n = len(self.channels)

        if n_zero == n:
            return 2  # 闭欧拉回路
        elif n_plus == 1 and n_minus == 1 and n_zero == n - 2:
            return 1  # 开欧拉迹
        else:
            return 0  # 非欧拉

    def coupling_connectivity(self) -> List[Set[int]]:
        """
        计算耦合图的连通分量 (BFS)

        返回:
            各连通分量的通道索引集合
        """
        n = len(self.channels)
        A = self.build_adjacency_matrix()
        visited = set()
        components = []

        for start in range(n):
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                # 找所有邻居 (有向图弱连通)
                for j in range(n):
                    if (abs(A[node, j]) > 1e-15 or abs(A[j, node]) > 1e-15) and j not in visited:
                        queue.append(j)
            if component:
                components.append(component)

        return components


class AngularMomentumCoupler:
    """
    角动量耦合组合分解器
    (映射自 905_pram — 拼图式组合枚举)

    对给定总角动量 J 和宇称 pi,
    枚举所有可能的 (l, j_channel) 组合

    耦合方案:
        J = l + j_channel  (向量耦合)
        pi = pi_target * (-1)^l  (宇称守恒)

    组合分解:
        对每个靶核态 J_t,
        入射粒子自旋 s_p,
        j_channel = J_t + s_p (矢量)
        l 满足 |J - j_channel| <= l <= J + j_channel
        且 (-1)^l = pi * pi_target
    """

    def __init__(
        self,
        J_total: float = 0.5,
        parity: int = 1,
        J_target: float = 0.0,
        s_projectile: float = 0.5,
        pi_target: int = 1,
    ):
        self.J_total = J_total
        self.parity = parity
        self.J_target = J_target
        self.s_projectile = s_projectile
        self.pi_target = pi_target

    def enumerate_channels(self) -> List[Dict]:
        """
        枚举所有允许的反应通道 (l, j_channel 组合)

        返回:
            通道参数列表
        """
        channels = []

        # j_channel 的可能值: |J_target - s_p| 到 J_target + s_p
        j_min = abs(self.J_target - self.s_projectile)
        j_max = self.J_target + self.s_projectile
        j_values = np.arange(j_min, j_max + 0.5, 1.0)

        for j_ch in j_values:
            # l 的可能值: |J - j_ch| 到 J + j_ch
            l_min = abs(self.J_total - j_ch)
            l_max = self.J_total + j_ch
            l_values = np.arange(l_min, l_max + 0.5, 1.0)

            for l_val in l_values:
                l_int = int(round(l_val))
                # 宇称检查
                channel_parity = self.pi_target * ((-1) ** l_int)
                if channel_parity != self.parity:
                    continue

                # Clebsch-Gordan 系数 (简化为统计因子)
                cg_weight = (2 * self.J_total + 1) / (
                    (2 * self.J_target + 1) * (2 * self.s_projectile + 1))

                channels.append({
                    'l': l_int,
                    'j_channel': float(j_ch),
                    'J_total': self.J_total,
                    'parity': channel_parity,
                    'cg_weight': cg_weight,
                    'label': f"l={l_int}_j={j_ch:.1f}",
                })

        return channels

    def coupling_matrix_dimension(self) -> int:
        """计算耦合矩阵维度 (通道数)"""
        return len(self.enumerate_channels())

    def coupling_topology(self) -> Dict:
        """
        分析耦合拓扑结构

        返回:
            通道数, 最大耦合度, 树/环结构信息
        """
        channels = self.enumerate_channels()
        n_channels = len(channels)

        # 可能的耦合对 (delta_l = 0, 2 的耦合为主)
        n_couplings = 0
        for i, ci in enumerate(channels):
            for j, cj in enumerate(channels):
                if i >= j:
                    continue
                dl = abs(ci['l'] - cj['l'])
                if dl in [0, 2]:  # 四极耦合为主
                    n_couplings += 1

        return {
            'n_channels': n_channels,
            'n_couplings': n_couplings,
            'coupling_density': (2.0 * n_couplings / (n_channels * (n_channels - 1))
                                 if n_channels > 1 else 0.0),
            'max_l': max(c['l'] for c in channels) if channels else 0,
        }
