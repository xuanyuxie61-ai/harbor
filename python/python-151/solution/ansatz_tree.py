"""
ansatz_tree.py
==============
自适应参数化量子电路的完全二叉树表示与圆弧参数化旋转

原项目映射:
- 1291_treepack: 完全二叉树遍历 (cbt_traverse)，图的树检测 (graph_adj_is_tree)
- 176_circle_arc_grid: 圆弧网格生成，用于参数化单量子比特旋转门的参数空间采样

科学功能:
本模块将量子电路的ansatz表示为一棵完全二叉树，其中每个节点
对应一个参数化单量子比特旋转门或双量子比特纠缠门。利用树遍历
算法实现自适应ansatz构建（类似ADAPT-VQE），并通过圆弧参数化
在Bloch球大圆上均匀采样旋转轴方向，优化参数初始化。
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class TreeNode:
    """完全二叉树节点，表示量子电路中的一个门操作。"""
    depth: int              # 节点在树中的深度
    index: int              # 节点在所在层的索引
    gate_type: str          # 'RX', 'RY', 'RZ', 'CNOT', 'ISWAP' 等
    qubits: Tuple[int, ...] # 作用的量子比特索引
    param: float = 0.0      # 门参数（对参数化门）
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    parent: Optional['TreeNode'] = None


class AnsatzTree:
    """
    基于完全二叉树的自适应ansatz表示。
    对应 1291_treepack 的 cbt_traverse 思想。
    """
    def __init__(self, n_qubits: int, max_depth: int = 4):
        self.n_qubits = n_qubits
        self.max_depth = max_depth
        self.root: Optional[TreeNode] = None
        self.leaf_nodes: List[TreeNode] = []
        self.parameters: np.ndarray = np.array([])
        self.param_count = 0

    def build_hardware_efficient(self, entangler: str = 'CNOT'):
        """
        构建硬件高效的启发式ansatz（Hardware Efficient Ansatz, HEA）。
        每层包含: RY(theta) 在每一量子比特 + 线性纠缠门链。
        """
        self.root = None
        self.leaf_nodes = []
        current_layer: List[TreeNode] = []

        for d in range(self.max_depth):
            layer_nodes = []
            # 单量子比特旋转门层
            for q in range(self.n_qubits):
                node = TreeNode(depth=d, index=q, gate_type='RY',
                                qubits=(q,), param=0.0)
                layer_nodes.append(node)
                if d == 0:
                    self.root = node
            # 纠缠门层
            for q in range(self.n_qubits - 1):
                node = TreeNode(depth=d, index=self.n_qubits + q,
                                gate_type=entangler, qubits=(q, q + 1))
                layer_nodes.append(node)
            current_layer = layer_nodes
            self.leaf_nodes.extend(layer_nodes)

        self._collect_parameters()

    def build_adaptive_layer(self, gradient_scores: np.ndarray,
                             operator_pool: List[str],
                             threshold: float = 1e-3):
        """
        自适应添加一层：根据梯度分数选择要添加的算符。
        这是ADAPT-VQE的核心思想。

        参数:
            gradient_scores: 各候选算符的梯度绝对值
            operator_pool: 候选算符池，如 ['Y0 X1', 'Z0 Y1', ...]
            threshold: 梯度阈值，低于此值停止增长
        """
        max_idx = int(np.argmax(gradient_scores))
        if gradient_scores[max_idx] < threshold:
            return False

        # 将选中的算符作为新的树层添加
        new_op = operator_pool[max_idx]
        # 解析Pauli字符串为旋转门序列（简化处理）
        qubits = [i for i, c in enumerate(new_op.replace(' ', '')) if c in 'XYZ']
        node = TreeNode(depth=self.max_depth, index=len(self.leaf_nodes),
                        gate_type='ADAPT', qubits=tuple(qubits), param=0.0)
        self.leaf_nodes.append(node)
        self.max_depth += 1
        self._collect_parameters()
        return True

    def _collect_parameters(self):
        """收集所有参数化节点的参数到向量中。"""
        params = []
        for node in self.leaf_nodes:
            if node.gate_type in ('RX', 'RY', 'RZ', 'ADAPT'):
                params.append(node.param)
        self.parameters = np.array(params, dtype=float)
        self.param_count = len(params)

    def set_parameters(self, params: np.ndarray):
        """从向量更新树中所有参数。"""
        if len(params) != self.param_count:
            raise ValueError(f"参数数量不匹配: {len(params)} != {self.param_count}")
        idx = 0
        for node in self.leaf_nodes:
            if node.gate_type in ('RX', 'RY', 'RZ', 'ADAPT'):
                node.param = params[idx]
                idx += 1
        self.parameters = params.copy()

    def traverse_inorder(self, callback: Callable[[TreeNode], None]):
        """中序遍历树，callback在每个节点调用。"""
        def _traverse(node):
            if node is None:
                return
            _traverse(node.left)
            callback(node)
            _traverse(node.right)
        _traverse(self.root)

    def evaluate_statevector(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算ansatz作用后的量子态（状态向量模拟）。
        仅支持小规模系统（n_qubits <= 10）。
        """
        dim = 2 ** self.n_qubits
        if initial_state is None:
            psi = np.zeros(dim, dtype=complex)
            psi[0] = 1.0
        else:
            psi = np.array(initial_state, dtype=complex)

        for node in self.leaf_nodes:
            if node.gate_type == 'RY':
                psi = _apply_ry(self.n_qubits, node.qubits[0], node.param, psi)
            elif node.gate_type == 'RX':
                psi = _apply_rx(self.n_qubits, node.qubits[0], node.param, psi)
            elif node.gate_type == 'RZ':
                psi = _apply_rz(self.n_qubits, node.qubits[0], node.param, psi)
            elif node.gate_type == 'CNOT':
                psi = _apply_cnot(self.n_qubits, node.qubits[0], node.qubits[1], psi)
            elif node.gate_type == 'ISWAP':
                psi = _apply_iswap(self.n_qubits, node.qubits[0], node.qubits[1], node.param, psi)
        return psi

    def get_circuit_depth(self) -> int:
        """返回电路深度。"""
        return self.max_depth


def _apply_rx(n: int, q: int, theta: float, psi: np.ndarray) -> np.ndarray:
    """应用单量子比特RX门: RX(theta) = exp(-i theta X / 2) = cos(theta/2) I - i sin(theta/2) X"""
    dim = 2 ** n
    psi_out = np.zeros(dim, dtype=complex)
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    stride = 2 ** q
    for i in range(dim):
        partner = i ^ stride
        if i < partner:
            psi_out[i] = c * psi[i] + s * psi[partner]
            psi_out[partner] = s * psi[i] + c * psi[partner]
    return psi_out


def _apply_ry(n: int, q: int, theta: float, psi: np.ndarray) -> np.ndarray:
    """应用单量子比特RY门: RY(theta) = exp(-i theta Y / 2) = cos(theta/2) I - i sin(theta/2) Y"""
    dim = 2 ** n
    psi_out = np.zeros(dim, dtype=complex)
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    stride = 2 ** q
    for i in range(dim):
        partner = i ^ stride
        if i < partner:
            if (i // stride) % 2 == 0:
                psi_out[i] = c * psi[i] - s * psi[partner]
                psi_out[partner] = s * psi[i] + c * psi[partner]
    return psi_out


def _apply_rz(n: int, q: int, theta: float, psi: np.ndarray) -> np.ndarray:
    """应用单量子比特RZ门: RZ(theta) = exp(-i theta Z / 2)"""
    dim = 2 ** n
    psi_out = np.zeros(dim, dtype=complex)
    stride = 2 ** q
    for i in range(dim):
        if (i // stride) % 2 == 0:
            psi_out[i] = np.exp(-1j * theta / 2.0) * psi[i]
        else:
            psi_out[i] = np.exp(1j * theta / 2.0) * psi[i]
    return psi_out


def _apply_cnot(n: int, control: int, target: int, psi: np.ndarray) -> np.ndarray:
    """应用CNOT门。"""
    dim = 2 ** n
    psi_out = psi.copy()
    c_stride = 2 ** control
    t_stride = 2 ** target
    for i in range(dim):
        if (i // c_stride) % 2 == 1:
            partner = i ^ t_stride
            if i < partner:
                psi_out[i], psi_out[partner] = psi[partner], psi[i]
    return psi_out


def _apply_iswap(n: int, q1: int, q2: int, theta: float, psi: np.ndarray) -> np.ndarray:
    """
    应用参数化iSWAP门:
        iSWAP(theta) = |00><00| + cos(theta)(|01><01| + |10><10|)
                       - i*sin(theta)(|01><10| + |10><01|) + |11><11|
    """
    dim = 2 ** n
    psi_out = psi.copy()
    s1 = 2 ** q1
    s2 = 2 ** q2
    c = np.cos(theta)
    s = -1j * np.sin(theta)
    for i in range(dim):
        b1 = (i // s1) % 2
        b2 = (i // s2) % 2
        if b1 == 1 and b2 == 0:
            partner = i ^ s1 ^ s2
            if i < partner:
                psi_out[i] = c * psi[i] + s * psi[partner]
                psi_out[partner] = s * psi[i] + c * psi[partner]
    return psi_out


def circle_arc_grid_params(r: float, center: np.ndarray,
                           angles_deg: Tuple[float, float],
                           n_points: int) -> np.ndarray:
    """
    在圆弧上均匀采样参数点，对应 176_circle_arc_grid/circle_arc_grid。

    在VQE中，用于在Bloch球的大圆上均匀初始化单量子比特旋转门的
    旋转轴方向参数。

    参数:
        r: 圆弧半径
        center: 圆心坐标 (2,) 或 (3,)
        angles_deg: 起始和终止角度（度）
        n_points: 采样点数
    返回:
        points: (n_points, dim) 参数点数组
    """
    center = np.asarray(center, dtype=float)
    dim = center.shape[0]
    if dim not in (2, 3):
        raise ValueError("仅支持2D或3D圆弧")
    angles = np.linspace(angles_deg[0], angles_deg[1], n_points)
    angles_rad = np.deg2rad(angles)
    points = np.zeros((n_points, dim))
    points[:, 0] = center[0] + r * np.cos(angles_rad)
    points[:, 1] = center[1] + r * np.sin(angles_rad)
    if dim == 3:
        points[:, 2] = center[2]
    return points


def initialize_parameters_on_bloch_circle(ansatz: AnsatzTree,
                                          r: float = np.pi,
                                          n_samples: int = 16):
    """
    在Bloch球赤道圆弧上均匀采样参数，用于初始化ansatz。
    对应 circle_arc_grid 在量子参数空间的应用。
    """
    n_params = ansatz.param_count
    if n_params == 0:
        return
    # 为每个参数在 [0, 2pi] 圆弧上均匀分配初始值
    points = circle_arc_grid_params(r, np.array([0.0, 0.0]),
                                    (0.0, 360.0), n_params)
    # 使用角度本身作为参数值，加入小幅随机扰动增强优化
    params = points[:, 0] + 0.1 * np.random.randn(n_params)
    ansatz.set_parameters(params)


def is_tree_adjacency(adj: np.ndarray) -> bool:
    """
    判断邻接矩阵是否表示一棵树，对应 1291_treepack/graph_adj_is_tree。
    在ansatz连通性分析中用于验证量子比特拓扑是否为树结构
    （如某些 trapped-ion 体系）。
    """
    nnode = adj.shape[0]
    if nnode <= 1:
        return True
    # 检查连通性（BFS）
    visited = np.zeros(nnode, dtype=bool)
    queue = [0]
    visited[0] = True
    while queue:
        v = queue.pop(0)
        for u in range(nnode):
            if adj[v, u] != 0 and not visited[u]:
                visited[u] = True
                queue.append(u)
    if not np.all(visited):
        return False
    # 检查边数是否为 n-1
    nedge = np.count_nonzero(adj) // 2
    return nedge == nnode - 1
