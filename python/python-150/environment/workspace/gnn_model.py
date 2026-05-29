"""
gnn_model.py
============
图神经网络模型

科学背景:
  本模型为物理信息增强的 Message Passing Neural Network (MPNN)，
  融合以下核心组件:
    - Chebyshev 谱图卷积 (chebyshev_conv.py)
    - 静电势特征 (electrostatic_solver.py)
    - 多项式描述符 (polynomial_basis.py)
    - 不确定性估计 (uncertainty_model.py)
    - 图分析工具 (graph_utils.py)

  消息传递方程:
      m_{ij}^{(l)} = MLP([h_i^{(l)}, h_j^{(l)}, e_{ij}])
      h_i^{(l+1)} = h_i^{(l)} + Σ_j m_{ij}^{(l)}

  其中 h_i^{(l)} 为第 l 层原子隐藏态，e_{ij} 为边特征（键级、距离、库仑势）。
"""

import numpy as np
from typing import List, Tuple


class MLP:
    """两层 MLP，用于消息/更新函数。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        limit1 = np.sqrt(6.0 / (in_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        limit2 = np.sqrt(6.0 / (hidden_dim + out_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, out_dim))
        self.b2 = np.zeros(out_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def parameters(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]


class MPNNLayer:
    """消息传递层。"""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        self.msg_mlp = MLP(2 * node_dim + edge_dim, hidden_dim, node_dim)
        self.update_mlp = MLP(node_dim, hidden_dim, node_dim)

    def __call__(self, node_features: np.ndarray, edge_list: List[Tuple[int, int]],
                 edge_features: np.ndarray) -> np.ndarray:
        n_nodes = node_features.shape[0]
        messages = np.zeros_like(node_features)
        for eidx, (i, j) in enumerate(edge_list):
            inp = np.concatenate([node_features[i], node_features[j], edge_features[eidx]])
            msg = self.msg_mlp(inp.reshape(1, -1)).flatten()
            messages[i] += msg
            messages[j] += msg  # 无向图双向消息
        updated = node_features + messages
        out = np.zeros_like(updated)
        for i in range(n_nodes):
            out[i] = self.update_mlp(updated[i].reshape(1, -1)).flatten()
        return out

    def parameters(self) -> list:
        return self.msg_mlp.parameters() + self.update_mlp.parameters()


class MolecularMPNN:
    """
    完整分子 MPNN，输出原子能、总能量、原子电荷及不确定性。
    """

    def __init__(self, node_in: int = 5, edge_in: int = 4,
                 hidden: int = 32, n_layers: int = 3):
        self.n_layers = n_layers
        self.layers: List[MPNNLayer] = []
        for _ in range(n_layers):
            self.layers.append(MPNNLayer(hidden, edge_in, hidden))

        # 初始投影
        limit = np.sqrt(6.0 / (node_in + hidden))
        self.W_init = np.random.uniform(-limit, limit, (node_in, hidden))
        self.b_init = np.zeros(hidden)

        # 输出头
        self.atom_energy_mlp = MLP(hidden, hidden, 1)
        self.atom_charge_mlp = MLP(hidden, hidden, 1)
        self.global_mlp = MLP(hidden, hidden, hidden)

        # 不确定性回归头
        from uncertainty_model import EvidentialRegressor
        self.uncertainty_head = EvidentialRegressor(hidden, hidden_dim=16)

    def _edge_features(self, graph, charges: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        边特征: [键级, 1/r, q_i*q_j/r, exp(-r)]。
        """
        edges = []
        feats = []
        for (a, b, order) in graph.bonds:
            r = np.linalg.norm(graph.atoms[a] - graph.atoms[b])
            r = max(r, 0.5)
            qiqj = charges[a] * charges[b] / r
            efeat = np.array([order, 1.0 / r, qiqj, np.exp(-r)], dtype=np.float64)
            edges.append((a, b))
            feats.append(efeat)
        if not edges:
            # 孤立原子虚边
            edges = [(0, 0)]
            feats = [np.zeros(4, dtype=np.float64)]
        return edges, np.array(feats, dtype=np.float64)

    def forward(self, graph, atomic_numbers: np.ndarray) -> dict:
        """
        前向传播。

        Returns
        -------
        dict with keys:
            atom_energies, total_energy, atom_charges,
            gamma, nu, alpha, beta (uncertainty params)
        """
        from feature_engineering import compute_atom_features
        from chebyshev_conv import ChebyshevGraphConv

        # 初始节点特征
        node_feats = compute_atom_features(atomic_numbers)  # (n, 5)
        h = np.maximum(0.0, node_feats @ self.W_init + self.b_init)  # (n, hidden)

        # Chebyshev 谱卷积前置滤波
        cheb = ChebyshevGraphConv(h.shape[1], h.shape[1], K=3)
        h = cheb(h, graph.apply_normalized_laplacian)

        # 边特征
        charges_init = atomic_numbers.astype(np.float64)
        edge_list, edge_feats = self._edge_features(graph, charges_init)

        # 消息传递
        for layer in self.layers:
            h = layer(h, edge_list, edge_feats)
            # LayerNorm (手工实现)
            mean = h.mean(axis=-1, keepdims=True)
            std = h.std(axis=-1, keepdims=True) + 1e-6
            h = (h - mean) / std

        # 原子级输出
        atom_energies = np.zeros(graph.n_atoms, dtype=np.float64)
        atom_charges = np.zeros(graph.n_atoms, dtype=np.float64)
        for i in range(graph.n_atoms):
            atom_energies[i] = self.atom_energy_mlp(h[i].reshape(1, -1)).flatten()[0]
            atom_charges[i] = self.atom_charge_mlp(h[i].reshape(1, -1)).flatten()[0]

        # 全局读出: sum pooling + MLP
        global_feat = self.global_mlp(h.sum(axis=0).reshape(1, -1)).flatten()

        # 总能量 = 原子能之和 + 全局修正
        total_energy = np.sum(atom_energies) + global_feat[0]

        # TODO: obtain uncertainty parameters from self.uncertainty_head.predict
        # and pack them into the returned dictionary together with other outputs.
        raise NotImplementedError("uncertainty parameter extraction is not implemented")

    def parameters(self) -> list:
        params = [self.W_init, self.b_init]
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.atom_energy_mlp.parameters())
        params.extend(self.atom_charge_mlp.parameters())
        params.extend(self.global_mlp.parameters())
        params.extend(self.uncertainty_head.parameters())
        return params
