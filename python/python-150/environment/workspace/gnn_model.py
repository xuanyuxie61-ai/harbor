
import numpy as np
from typing import List, Tuple


class MLP:

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
            messages[j] += msg
        updated = node_features + messages
        out = np.zeros_like(updated)
        for i in range(n_nodes):
            out[i] = self.update_mlp(updated[i].reshape(1, -1)).flatten()
        return out

    def parameters(self) -> list:
        return self.msg_mlp.parameters() + self.update_mlp.parameters()


class MolecularMPNN:

    def __init__(self, node_in: int = 5, edge_in: int = 4,
                 hidden: int = 32, n_layers: int = 3):
        self.n_layers = n_layers
        self.layers: List[MPNNLayer] = []
        for _ in range(n_layers):
            self.layers.append(MPNNLayer(hidden, edge_in, hidden))


        limit = np.sqrt(6.0 / (node_in + hidden))
        self.W_init = np.random.uniform(-limit, limit, (node_in, hidden))
        self.b_init = np.zeros(hidden)


        self.atom_energy_mlp = MLP(hidden, hidden, 1)
        self.atom_charge_mlp = MLP(hidden, hidden, 1)
        self.global_mlp = MLP(hidden, hidden, hidden)


        from uncertainty_model import EvidentialRegressor
        self.uncertainty_head = EvidentialRegressor(hidden, hidden_dim=16)

    def _edge_features(self, graph, charges: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
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

            edges = [(0, 0)]
            feats = [np.zeros(4, dtype=np.float64)]
        return edges, np.array(feats, dtype=np.float64)

    def forward(self, graph, atomic_numbers: np.ndarray) -> dict:
        from feature_engineering import compute_atom_features
        from chebyshev_conv import ChebyshevGraphConv


        node_feats = compute_atom_features(atomic_numbers)
        h = np.maximum(0.0, node_feats @ self.W_init + self.b_init)


        cheb = ChebyshevGraphConv(h.shape[1], h.shape[1], K=3)
        h = cheb(h, graph.apply_normalized_laplacian)


        charges_init = atomic_numbers.astype(np.float64)
        edge_list, edge_feats = self._edge_features(graph, charges_init)


        for layer in self.layers:
            h = layer(h, edge_list, edge_feats)

            mean = h.mean(axis=-1, keepdims=True)
            std = h.std(axis=-1, keepdims=True) + 1e-6
            h = (h - mean) / std


        atom_energies = np.zeros(graph.n_atoms, dtype=np.float64)
        atom_charges = np.zeros(graph.n_atoms, dtype=np.float64)
        for i in range(graph.n_atoms):
            atom_energies[i] = self.atom_energy_mlp(h[i].reshape(1, -1)).flatten()[0]
            atom_charges[i] = self.atom_charge_mlp(h[i].reshape(1, -1)).flatten()[0]


        global_feat = self.global_mlp(h.sum(axis=0).reshape(1, -1)).flatten()


        total_energy = np.sum(atom_energies) + global_feat[0]



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
