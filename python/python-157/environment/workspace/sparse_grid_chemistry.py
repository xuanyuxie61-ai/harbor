"""
sparse_grid_chemistry.py
高维化学流形稀疏网格插值模块
融合来源：1133_spinterp（稀疏网格分层插值）

用于在高维参数空间（温度、压强、当量比、残余气体分数等）
中对复杂化学反应速率进行快速近似评估。
"""
import numpy as np
from combustion_utils import check_positive, check_nonnegative, check_interval


def clenshaw_curtis_nodes_1d(level):
    r"""
    生成一维 Clenshaw-Curtis 节点:
        x_j = cos(pi * j / (2^l)), j = 0, ..., 2^l
    对应 level l 的节点数为 n = 2^l + 1。
    level >= 0。
    """
    check_nonnegative(level, "level")
    if level == 0:
        return np.array([0.0])
    n = 2 ** level
    j = np.arange(n + 1)
    x = np.cos(np.pi * j / n)
    return x


def piecewise_linear_basis(nodes, x_eval):
    r"""
    在节点集 nodes（已排序）上计算 x_eval 处的分段线性基函数值。
    返回权重向量 w，使得 f(x_eval) ≈ Σ w_i * f(nodes_i)。
    """
    nodes = np.asarray(nodes)
    x = float(x_eval)
    n = len(nodes)
    w = np.zeros(n)
    if x <= nodes[0]:
        w[0] = 1.0
        return w
    if x >= nodes[-1]:
        w[-1] = 1.0
        return w
    # 找到包含 x 的区间
    for i in range(n - 1):
        if nodes[i] <= x <= nodes[i + 1]:
            h = nodes[i + 1] - nodes[i]
            if abs(h) < 1.0e-14:
                w[i] = 1.0
            else:
                w[i] = (nodes[i + 1] - x) / h
                w[i + 1] = (x - nodes[i]) / h
            return w
    w[-1] = 1.0
    return w


def sparse_grid_index_set(dim, max_level):
    r"""
    生成稀疏网格索引集:
        I = { l ∈ N^d : |l|_1 <= max_level + d - 1 }
    其中 l_i >= 1 为各维度的层数。
    """
    indices = []

    def recurse(current, dim_idx, sum_level):
        if dim_idx == dim:
            if sum_level <= max_level + dim - 1:
                indices.append(current[:])
            return
        for l in range(1, max_level + 1):
            if sum_level + l > max_level + dim - 1:
                break
            current.append(l)
            recurse(current, dim_idx + 1, sum_level + l)
            current.pop()

    recurse([], 0, 0)
    return indices


class SparseGridChemistry:
    r"""
    高维化学反应速率稀疏网格近似器。

    维度示例:
        dim1: 归一化温度   T/T_max ∈ [0, 1]
        dim2: 归一化压强   p/p_max ∈ [0, 1]
        dim3: 当量比       phi ∈ [0.5, 2.0] 归一化
        dim4: 残余气体分数 x_r ∈ [0, 1]
    """

    def __init__(self, max_level=3, dim=4):
        check_positive(max_level, "max_level")
        check_positive(dim, "dim")
        self.max_level = max_level
        self.dim = dim
        self.grids = [{} for _ in range(dim)]
        self.values = {}

    def build(self, rate_func):
        r"""
        构建稀疏网格，对 rate_func 在各层级节点上求值。
        rate_func(y) 接受 d 维归一化向量，返回标量反应速率。
        """
        # 对每个维度独立构建分层数据结构
        for di in range(self.dim):
            for li in range(0, self.max_level + 1):
                nodes = clenshaw_curtis_nodes_1d(li)
                self.grids[di][li] = nodes

        # 在稀疏网格节点上求值
        self.values = {}
        for level_vec in sparse_grid_index_set(self.dim, self.max_level):
            nodes_list = [self.grids[d][level_vec[d]] for d in range(self.dim)]
            # 张量积节点
            mesh = np.meshgrid(*nodes_list, indexing='ij')
            flat_nodes = np.stack([m.ravel() for m in mesh], axis=1)
            for pt in flat_nodes:
                key = tuple(np.round(pt, decimals=12))
                if key not in self.values:
                    self.values[key] = rate_func(pt)

    def evaluate(self, y_point):
        r"""
        在 y_point 处评估插值近似值。
        y_point: ndarray, shape (dim,)，各分量应在 [-1, 1] 内。
        """
        y_point = np.asarray(y_point, dtype=float)
        if y_point.shape[0] != self.dim:
            raise ValueError(f"Point dimension {y_point.shape[0]} != grid dimension {self.dim}")
        y_point = np.clip(y_point, -1.0, 1.0)

        # 使用逐维线性插值
        result = 0.0
        count = 0
        for level_vec in sparse_grid_index_set(self.dim, self.max_level):
            nodes_list = [self.grids[d][level_vec[d]] for d in range(self.dim)]
            mesh = np.meshgrid(*nodes_list, indexing='ij')
            flat_nodes = np.stack([m.ravel() for m in mesh], axis=1)
            flat_vals = np.array([self.values.get(tuple(np.round(pt, decimals=12)), 0.0) for pt in flat_nodes])

            # 对每维做线性插值
            interp_val = flat_vals.reshape([len(nl) for nl in nodes_list])
            for d in range(self.dim):
                nodes_d = nodes_list[d]
                w = piecewise_linear_basis(nodes_d, y_point[d])
                # 沿维度 d 加权求和
                shape = list(interp_val.shape)
                shape[d] = 1
                new_val = np.zeros(shape)
                for i in range(len(nodes_d)):
                    slc = [slice(None)] * self.dim
                    slc[d] = i
                    new_val += w[i] * interp_val[tuple(slc)]
                interp_val = new_val
            result += interp_val.flat[0]
            count += 1

        if count > 0:
            result /= count
        return result

    def evaluate_batch(self, points):
        r"""
        批量评估，points 形状为 (n_points, dim)。
        """
        points = np.asarray(points, dtype=float)
        return np.array([self.evaluate(p) for p in points])
