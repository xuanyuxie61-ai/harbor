
import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class TreeNode:
    depth: int
    index: int
    gate_type: str
    qubits: Tuple[int, ...]
    param: float = 0.0
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    parent: Optional['TreeNode'] = None


class AnsatzTree:
    def __init__(self, n_qubits: int, max_depth: int = 4):
        self.n_qubits = n_qubits
        self.max_depth = max_depth
        self.root: Optional[TreeNode] = None
        self.leaf_nodes: List[TreeNode] = []
        self.parameters: np.ndarray = np.array([])
        self.param_count = 0

    def build_hardware_efficient(self, entangler: str = 'CNOT'):
        self.root = None
        self.leaf_nodes = []
        current_layer: List[TreeNode] = []

        for d in range(self.max_depth):
            layer_nodes = []

            for q in range(self.n_qubits):
                node = TreeNode(depth=d, index=q, gate_type='RY',
                                qubits=(q,), param=0.0)
                layer_nodes.append(node)
                if d == 0:
                    self.root = node

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
        max_idx = int(np.argmax(gradient_scores))
        if gradient_scores[max_idx] < threshold:
            return False


        new_op = operator_pool[max_idx]

        qubits = [i for i, c in enumerate(new_op.replace(' ', '')) if c in 'XYZ']
        node = TreeNode(depth=self.max_depth, index=len(self.leaf_nodes),
                        gate_type='ADAPT', qubits=tuple(qubits), param=0.0)
        self.leaf_nodes.append(node)
        self.max_depth += 1
        self._collect_parameters()
        return True

    def _collect_parameters(self):
        params = []
        for node in self.leaf_nodes:
            if node.gate_type in ('RX', 'RY', 'RZ', 'ADAPT'):
                params.append(node.param)
        self.parameters = np.array(params, dtype=float)
        self.param_count = len(params)

    def set_parameters(self, params: np.ndarray):
        if len(params) != self.param_count:
            raise ValueError(f"参数数量不匹配: {len(params)} != {self.param_count}")
        idx = 0
        for node in self.leaf_nodes:
            if node.gate_type in ('RX', 'RY', 'RZ', 'ADAPT'):
                node.param = params[idx]
                idx += 1
        self.parameters = params.copy()

    def traverse_inorder(self, callback: Callable[[TreeNode], None]):
        def _traverse(node):
            if node is None:
                return
            _traverse(node.left)
            callback(node)
            _traverse(node.right)
        _traverse(self.root)

    def evaluate_statevector(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
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
        return self.max_depth


def _apply_rx(n: int, q: int, theta: float, psi: np.ndarray) -> np.ndarray:
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
    n_params = ansatz.param_count
    if n_params == 0:
        return

    points = circle_arc_grid_params(r, np.array([0.0, 0.0]),
                                    (0.0, 360.0), n_params)

    params = points[:, 0] + 0.1 * np.random.randn(n_params)
    ansatz.set_parameters(params)


def is_tree_adjacency(adj: np.ndarray) -> bool:
    nnode = adj.shape[0]
    if nnode <= 1:
        return True

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

    nedge = np.count_nonzero(adj) // 2
    return nedge == nnode - 1
