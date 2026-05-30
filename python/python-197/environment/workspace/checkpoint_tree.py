
import numpy as np


class CheckpointNode:

    def __init__(self, level: int, name: str, write_bw: float, read_bw: float,
                 capacity: float, cost_per_gb: float):
        self.level = level
        self.name = name
        self.write_bw = write_bw
        self.read_bw = read_bw
        self.capacity = capacity
        self.cost_per_gb = cost_per_gb
        self.children = []
        self.parent = None
        self.stored_checkpoints = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def write_time(self, data_gb: float) -> float:
        if self.write_bw <= 0.0:
            return 1.0e10
        return data_gb / self.write_bw

    def read_time(self, data_gb: float) -> float:
        if self.read_bw <= 0.0:
            return 1.0e10
        return data_gb / self.read_bw

    def storage_cost(self, data_gb: float) -> float:
        return data_gb * self.cost_per_gb


class CheckpointTree:

    def __init__(self, root: CheckpointNode):
        self.root = root
        self._nodes = []
        self._collect_nodes(root)

    def _collect_nodes(self, node: CheckpointNode):
        self._nodes.append(node)
        for c in node.children:
            self._collect_nodes(c)

    def get_level(self, level: int):
        return [n for n in self._nodes if n.level == level]

    def expected_recovery_time(self, data_gb: float, level_probs: dict) -> float:
        ert = 0.0
        for node in self._nodes:
            p = level_probs.get(node.level, 0.0)
            if p <= 0.0:
                continue
            t = node.read_time(data_gb)

            cur = node
            while cur.parent is not None:
                t += cur.parent.read_time(data_gb)
                cur = cur.parent
            ert += p * t
        return ert

    def total_storage_cost(self, data_gb: float, replication_levels: list) -> float:
        cost = 0.0
        for node in self._nodes:
            if node.level in replication_levels:
                cost += node.storage_cost(data_gb)
        return cost

    def tree_distance(self, node_a: CheckpointNode, node_b: CheckpointNode) -> int:
        if node_a is node_b:
            return 0

        path_a = []
        cur = node_a
        while cur is not None:
            path_a.append(cur)
            cur = cur.parent
        path_b = []
        cur = node_b
        while cur is not None:
            path_b.append(cur)
            cur = cur.parent

        common = None
        for a in path_a:
            if a in path_b:
                common = a
                break
        if common is None:
            return len(path_a) + len(path_b)
        dist = 0
        cur = node_a
        while cur is not common:
            cur = cur.parent
            dist += 1
        cur = node_b
        while cur is not common:
            cur = cur.parent
            dist += 1
        return dist

    def hierarchical_cluster_placement(self, n_checkpoints: int) -> dict:
        leaves = [n for n in self._nodes if len(n.children) == 0]
        if not leaves:
            leaves = [self.root]
        counts = {leaf.name: 0 for leaf in leaves}

        for i in range(n_checkpoints):
            counts[leaves[i % len(leaves)].name] += 1
        return counts


def build_default_tree():
    mem = CheckpointNode(0, "DRAM", write_bw=10.0, read_bw=20.0,
                         capacity=64.0, cost_per_gb=100.0)
    ssd = CheckpointNode(1, "Local_SSD", write_bw=2.0, read_bw=3.0,
                         capacity=2000.0, cost_per_gb=10.0)
    pfs = CheckpointNode(2, "Remote_PFS", write_bw=0.5, read_bw=0.8,
                         capacity=1.0e6, cost_per_gb=1.0)
    mem.add_child(ssd)
    ssd.add_child(pfs)
    return CheckpointTree(mem)
