"""
free_energy_integration.py
===========================
基于高维稀疏网格与多变量求积的结合自由能计算模块。

核心数学内容：
  - 总阶数稀疏网格（Total Polynomial Sparse Grid）：
    对 $d$ 维空间，层级 $L$ 的稀疏网格为
      $H(L,d) = \bigcup_{|\ell|_1 \le L} X_{\ell_1} \times \cdots \times X_{\ell_d}$
    其中 $|\ell|_1 = \ell_1 + \cdots + \ell_d$，$X_{\ell_i}$ 为第 $i$ 维的 1D  nested 节点集。
  - 结合自由能的热力学积分（Thermodynamic Integration, TI）：
      $\Delta G_{\text{bind}} = \int_0^1 \left\langle \frac{\partial H(\lambda)}{\partial \lambda} \right\rangle_\lambda \, d\lambda$
    其中 $H(\lambda) = H_0 + \lambda H_{\text{int}}$ 为耦合参数哈密顿量。
  - Clenshaw-Curtis、Jacobi、Laguerre 多重量子求积。

种子项目映射：
  - 1109_sparse_grid_total_poly  →  稀疏网格构造、层级索引、尺寸计算
  - 1054_sandia_rules           →  各类 1D 求积规则
  - 1324_triangle_wandzura_rule →  三角形面积分（膜表面自由能）
"""

import numpy as np
from typing import Callable, List, Tuple
from quadrature_rules import clenshaw_curtis_compute, jacobi_compute, laguerre_quadrature_rule
from quadrature_rules import integrate_triangle


# ---------------------------------------------------------------------------
# 组合辅助函数（种子项目 1109_sparse_grid_total_poly / comp_next）
# ---------------------------------------------------------------------------
def comp_next(n: int, k: int, a: np.ndarray, more: bool, h: int, t: int) -> Tuple[np.ndarray, bool, int, int]:
    """
    生成整数 $n$ 的 $k$ 部分组合（compositions）。

    算法来源：Albert Nijenhuis, Herbert Wilf,
    "Combinatorial Algorithms for Computers and Calculators", 1978.
    原始为 1-based 索引，此处转换为 0-based Python 索引。

    参数：
        n     : 总和
        k     : 部分数
        a     : 当前组合（长度 k）
        more  : 是否还有更多组合
        h, t  : 内部状态

    返回：
        a, more, h, t
    """
    if not more:
        a = np.zeros(k, dtype=int)
        a[0] = n
        h = 0
        t = n
        more = (a[-1] != n)
        return a, more, h, t

    if 1 < t:
        h = 0
    h += 1

    # 边界保护：h 不应超过 k-1（对应 MATLAB 中的 k）
    if h >= k:
        more = False
        return a, more, h, t

    # MATLAB 原始代码（1-based）：
    #   t = a(h); a(h) = 0; a(1) = t - 1; a(h+1) = a(h+1) + 1;
    # 转换为 0-based Python：
    t_val = a[h - 1]
    a[h - 1] = 0
    a[0] = t_val - 1
    a[h] += 1
    t = t_val

    more = (a[-1] != n)
    return a, more, h, t


def level_to_order_closed(level: int) -> int:
    """
    将 Clenshaw-Curtis 1D 层级映射为节点数。
    层级 0 -> 1 点，层级 1 -> 3 点，层级 l -> $2^l + 1$ 点。
    """
    if level == 0:
        return 1
    return 2 ** level + 1


# ---------------------------------------------------------------------------
# 稀疏网格尺寸计算（种子项目 1109_sparse_grid_total_poly / sparse_grid_total_poly_size）
# ---------------------------------------------------------------------------
def sparse_grid_total_poly_size(dim_num: int, level_max: int) -> int:
    """
    计算总阶数稀疏网格的唯一点数。

    数学内容：
      对 nested 规则，1D 层级 $\ell$ 新增的点数为：
        new_1d[0] = 1, new_1d[1] = 2, new_1d[l] = 2^{l-1} (l >= 2)
      总点数 = $\sum_{|\ell|_1 \le L} \prod_{i=1}^{d} \text{new_1d}[\ell_i]$

    参数边界：
        dim_num >= 1, level_max >= 0
    """
    if dim_num < 1:
        raise ValueError("sparse_grid_total_poly_size: dim_num must be >= 1.")
    if level_max < 0:
        return 0
    if level_max == 0:
        return 1

    new_1d = np.zeros(level_max + 1, dtype=int)
    new_1d[0] = 1
    if level_max >= 1:
        new_1d[1] = 2
    for l in range(2, level_max + 1):
        new_1d[l] = 2 ** (l - 1)

    point_num = 0
    for level in range(level_max + 1):
        level_1d = np.zeros(dim_num, dtype=int)
        more = False
        h = 0
        t = 0
        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)
            point_num += int(np.prod(new_1d[level_1d]))
            if not more:
                break

    return point_num


# ---------------------------------------------------------------------------
# 稀疏网格索引构造（种子项目 1109_sparse_grid_total_poly / sparse_grid_total_poly_index）
# ---------------------------------------------------------------------------
def sparse_grid_total_poly_index(dim_num: int, level_max: int) -> np.ndarray:
    """
    构造稀疏网格的唯一点索引集合。

    返回：
        grid_index : shape (dim_num, point_num) 的整数数组，
                     每列是一个网格点的多索引。
    """
    point_num = sparse_grid_total_poly_size(dim_num, level_max)
    if point_num == 0:
        return np.zeros((dim_num, 0), dtype=int)

    new_1d = np.zeros(level_max + 1, dtype=int)
    new_1d[0] = 1
    if level_max >= 1:
        new_1d[1] = 2
    for l in range(2, level_max + 1):
        new_1d[l] = 2 ** (l - 1)

    # 先收集所有点（可能有重复）
    all_indices = []
    for level in range(level_max + 1):
        level_1d = np.zeros(dim_num, dtype=int)
        more = False
        h = 0
        t = 0
        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)
            # 构造该层级组合的乘积网格
            orders = np.array([level_to_order_closed(ld) for ld in level_1d], dtype=int)
            # 仅新增点（nested 规则下，层级 l 的新增点位于奇数位置）
            # 为简化，我们生成所有点然后去重
            prod_points = int(np.prod(orders))
            idx_array = np.zeros((dim_num, prod_points), dtype=int)
            # 用笛卡尔积生成索引
            grids = [np.arange(o) for o in orders]
            mesh = np.array(np.meshgrid(*grids, indexing='ij'))
            idx_array = mesh.reshape(dim_num, -1)
            all_indices.append(idx_array)
            if not more:
                break

    if not all_indices:
        return np.zeros((dim_num, 0), dtype=int)

    combined = np.hstack(all_indices)
    # 去重
    unique_cols = np.unique(combined, axis=1)
    return unique_cols


# ---------------------------------------------------------------------------
# 稀疏网格积分
# ---------------------------------------------------------------------------
def sparse_grid_integrate(
    dim_num: int,
    level_max: int,
    f: Callable[[np.ndarray], np.ndarray],
    rule: str = "clenshaw-curtis",
) -> float:
    """
    使用稀疏网格对 $d$ 维函数 $f$ 在 $[-1,1]^d$ 上求积。

    参数：
        dim_num   : 维度数，>= 1
        level_max : 最大层级，>= 0
        f         : 函数，接受 shape (dim_num, n) 返回 shape (n,)
        rule      : "clenshaw-curtis" | "jacobi" | "laguerre"

    参数边界：
        dim_num >= 1, level_max >= 0
    """
    if dim_num < 1:
        raise ValueError("sparse_grid_integrate: dim_num must be >= 1.")
    if level_max < 0:
        return 0.0

    # 生成 1D 规则到所需最大阶数
    max_order = level_to_order_closed(level_max)
    if rule == "clenshaw-curtis":
        x_1d, w_1d = clenshaw_curtis_compute(max_order)
    elif rule == "jacobi":
        x_1d, w_1d = jacobi_compute(max_order, 0.0, 0.0)
    elif rule == "laguerre":
        x_1d, w_1d = laguerre_quadrature_rule(max_order)
        # Laguerre 在 [0, inf)，需要映射；这里简化为直接使用
    else:
        raise ValueError("sparse_grid_integrate: unknown rule.")

    # 使用稀疏网格组合求积（简化实现：使用全张量积到 level_max，再截断）
    # 对于小维度小层级，直接用全积
    if dim_num <= 3 and level_max <= 4:
        orders = [level_to_order_closed(level_max)] * dim_num
        grids_x = [x_1d[:o] for o in orders]
        grids_w = [w_1d[:o] for o in orders]

        mesh = np.array(np.meshgrid(*grids_x, indexing='ij'))
        points = mesh.reshape(dim_num, -1)

        weight_mesh = np.array(np.meshgrid(*grids_w, indexing='ij'))
        weights = np.prod(weight_mesh.reshape(dim_num, -1), axis=0)

        vals = f(points)
        result = float(np.dot(weights, vals))
        return result

    # 大维度使用稀疏网格组合
    grid_index = sparse_grid_total_poly_index(dim_num, level_max)
    n_pts = grid_index.shape[1]
    points = np.zeros((dim_num, n_pts), dtype=float)
    weights = np.ones(n_pts, dtype=float)

    for d in range(dim_num):
        max_o = level_to_order_closed(level_max)
        points[d, :] = x_1d[grid_index[d, :] % max_o]
        weights *= w_1d[grid_index[d, :] % max_o]

    vals = f(points)
    result = float(np.dot(weights, vals))
    return result


# ---------------------------------------------------------------------------
# 热力学积分（Thermodynamic Integration）
# ---------------------------------------------------------------------------
def thermodynamic_integration_binding_free_energy(
    n_lambda: int = 11,
    temperature: float = 300.0,  # K
    dim_conformational: int = 3,
    sg_level: int = 3,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    使用热力学积分计算药物-蛋白结合自由能。

    物理模型：
      $\Delta G_{\text{bind}} = \int_0^1 \langle \partial_\lambda U(\lambda) \rangle_\lambda \, d\lambda$

    耦合哈密顿量：
      $U(\lambda) = U_{\text{protein}} + U_{\text{drug}} + \lambda U_{\text{int}}$

    其中 $U_{\text{int}}$ 包含 Lennard-Jones 与静电相互作用。

    积分方法：
      对 $\lambda$ 使用 Clenshaw-Curtis 求积；
      对每个 $\lambda$，对构象空间使用稀疏网格积分估计系综平均。

    参数边界：
        n_lambda >= 2, temperature > 0, dim_conformational >= 1, sg_level >= 0
    """
    if n_lambda < 2:
        raise ValueError("thermodynamic_integration_binding_free_energy: n_lambda >= 2.")
    if temperature <= 0:
        raise ValueError("thermodynamic_integration_binding_free_energy: temperature > 0.")
    if dim_conformational < 1:
        raise ValueError("thermodynamic_integration_binding_free_energy: dim_conformational >= 1.")

    # TODO: Hole 2 — 实现热力学积分的核心计算
    # 1. 定义 kB 和 beta
    # 2. 使用 clenshaw_curtis_compute 获取 lambda 节点与权重，并映射到 [0,1]
    # 3. 对每个 lambda，定义被积函数（含 Boltzmann 权重）和配分函数被积函数
    # 4. 使用 sparse_grid_integrate 计算分子（<dU/dλ> 的加权平均）和分母（配分函数）
    # 5. 计算 delta_G = sum(lam_weights * dU_dlambda)
    # 注意：lambda 映射必须与 Clenshaw-Curtis 节点范围 [-1,1] 一致
    raise NotImplementedError("Hole 2: Thermodynamic integration core not implemented.")


# ---------------------------------------------------------------------------
# 膜表面自由能（三角形积分）
# ---------------------------------------------------------------------------
def membrane_surface_free_energy(
    triangles: List[np.ndarray],
    energy_density: Callable[[np.ndarray], np.ndarray],
    rule_index: int = 2,
) -> float:
    """
    在膜表面的三角网格上积分表面自由能密度。

    数学形式：
        $G_{\text{surf}} = \sum_{T} \int_T \gamma(x,y) \, dA$
    其中 $\gamma(x,y)$ 为局部表面张力（包含疏水效应、曲率能等）。

    参数边界：
        triangles    : 三角形顶点列表，每个元素 shape (3, 2)
        energy_density: 函数，接受 shape (n, 2) 返回 shape (n,)
        rule_index   : 1 (6阶) 或 2 (12阶)
    """
    total = 0.0
    for tri in triangles:
        total += integrate_triangle(energy_density, tri, rule_index)
    return total
