"""
optimal_harvest.py
最优捕捞策略与海洋保护区网络优化模块

整合算法：
1. Brent 反向通信局部优化（基于 local_min_rc）：求解最优捕捞努力量 E*
2. Bellman-Ford 图算法：海洋保护区（MPA）网络最短路径与连通性优化

核心科学问题：
在考虑种群动态、经济贴现和生态约束的条件下，
寻找使长期贴现经济收益最大化的最优捕捞努力量 E*(t)

数学模型：
1. 最优控制目标泛函（Pontryagin 最大值原理）：
   J(E) = \int_0^T e^{-\delta t} [p q E(t) B(t) - c E(t)] dt
   其中 B(t) 为生物量，q 为可捕系数，p 为鱼价，c 为单位成本，\delta 为贴现率

2. 生物量动态（Schaefer-Gordon 模型）：
   dB/dt = r B (1 - B/K) - q E B
   r: 内禀增长率, K: 环境承载力

3. 稳态最优捕捞（MSY 框架）：
   E_{MSY} = r / (2q),  Y_{MSY} = rK / 4

4. MPA 网络优化：
   将各渔区抽象为图节点，边权重为生态连通性成本
   利用 Bellman-Ford 求解最短路径以优化资源调配
"""

import numpy as np
from utils import NumericalConfig, safe_divide


class BrentOptimizer:
    """
    Brent 方法局部优化器（反向通信模式改写为类封装）
    结合黄金分割搜索与抛物线插值

    算法流程：
    1. 初始化黄金分割常数 c = (3 - sqrt(5)) / 2 ≈ 0.381966
    2. 迭代更新区间 [a,b]，维护点 x,w,v 及其函数值 fx,fw,fv
    3. 优先尝试抛物线插值：拟合 (x,fx),(w,fw),(v,fv) 的抛物线
    4. 若抛物线最小值在区间内且足够有效，则采用；否则用黄金分割步
    5. 收敛准则：|x - midpoint| <= tol2 - 0.5*(b-a)
    """

    def __init__(self, a, b, tol=None):
        if b <= a:
            raise ValueError("Require a < b for Brent optimizer")
        self.a = float(a)
        self.b = float(b)
        self.c = 0.5 * (3.0 - np.sqrt(5.0))  # 黄金分割常数平方的逆
        self.eps_sqrt = NumericalConfig.EPS_SQRT
        self.tol = tol if tol is not None else NumericalConfig.TOL

        self.v = self.a + self.c * (self.b - self.a)
        self.w = self.v
        self.x = self.v
        self.e = 0.0
        self.fx = None
        self.fv = None
        self.fw = None
        self.status = 1
        self.arg = self.x
        self._first_call = True

    def step(self, value):
        """
        执行一次 Brent 迭代步

        Parameters
        ----------
        value : float
            上一步请求点处的函数值

        Returns
        -------
        arg : float
            下一个需要评估函数值的点
        status : int
            0 表示收敛，>0 表示继续迭代
        """
        if self._first_call:
            self.fx = value
            self.fv = self.fx
            self.fw = self.fx
            self._first_call = False
            self.status = 2
            self.arg = self.x
            return self.arg, self.status

        fu = value

        if fu <= self.fx:
            if self.x <= self.arg:
                self.a = self.x
            else:
                self.b = self.x
            self.v = self.w
            self.fv = self.fw
            self.w = self.x
            self.fw = self.fx
            self.x = self.arg
            self.fx = fu
        else:
            if self.arg < self.x:
                self.a = self.arg
            else:
                self.b = self.arg
            if fu <= self.fw or self.w == self.x:
                self.v = self.w
                self.fv = self.fw
                self.w = self.arg
                self.fw = fu
            elif fu <= self.fv or self.v == self.x or self.v == self.w:
                self.v = self.arg
                self.fv = fu

        midpoint = 0.5 * (self.a + self.b)
        tol1 = self.eps_sqrt * abs(self.x) + self.tol / 3.0
        tol2 = 2.0 * tol1

        if abs(self.x - midpoint) <= (tol2 - 0.5 * (self.b - self.a)):
            self.status = 0
            self.arg = self.x
            return self.arg, self.status

        if abs(self.e) <= tol1:
            if midpoint <= self.x:
                self.e = self.a - self.x
            else:
                self.e = self.b - self.x
            d = self.c * self.e
        else:
            r_val = (self.x - self.w) * (self.fx - self.fv)
            q_val = (self.x - self.v) * (self.fx - self.fw)
            p_val = (self.x - self.v) * q_val - (self.x - self.w) * r_val
            q_val = 2.0 * (q_val - r_val)
            if 0.0 < q_val:
                p_val = -p_val
            q_val = abs(q_val)
            r_val = self.e
            self.e = d if hasattr(self, 'd') else 0.0

            if (abs(0.5 * q_val * r_val) <= abs(p_val)) and (p_val > q_val * (self.a - self.x)) and (p_val < q_val * (self.b - self.x)):
                d = p_val / q_val
                u = self.x + d
                if (u - self.a) < tol2:
                    d = tol1 * np.sign(midpoint - self.x)
                if (self.b - u) < tol2:
                    d = tol1 * np.sign(midpoint - self.x)
            else:
                if midpoint <= self.x:
                    self.e = self.a - self.x
                else:
                    self.e = self.b - self.x
                d = self.c * self.e

        if abs(d) >= tol1:
            u = self.x + d
        else:
            u = self.x + tol1 * np.sign(d)

        self.arg = u
        self.status += 1
        return self.arg, self.status

    def optimize(self, func):
        """
        便捷接口：直接对 func 进行优化

        Parameters
        ----------
        func : callable
            一元标量函数

        Returns
        -------
        x_opt : float
            最优解
        f_opt : float
            最优函数值
        """
        arg, status = self.arg, self.status
        while status > 0:
            val = func(arg)
            arg, status = self.step(val)
        return arg, func(arg)


def schaefer_gordon_steady_state(E, r, K, q):
    """
    Schaefer-Gordon 生物经济模型的稳态生物量

    方程：dB/dt = rB(1 - B/K) - qEB = 0
    解得：B*(E) = K (1 - qE/r)  当 qE <= r
          B*(E) = 0              当 qE > r

    Parameters
    ----------
    E : float or ndarray
        捕捞努力量
    r : float
        内禀增长率
    K : float
        环境承载力
    q : float
        可捕系数

    Returns
    -------
    B : float or ndarray
        稳态生物量
    """
    E = np.asarray(E, dtype=float)
    B = K * (1.0 - q * E / r)
    B = np.where(q * E <= r, B, 0.0)
    return np.maximum(B, 0.0)


def discounted_profit_objective(E, r, K, q, p, c, delta, T=50.0):
    """
    计算贴现总利润（作为优化目标函数的负值，用于最小化）

    在稳态假设下，长期贴现利润为：
        \Pi(E) = \int_0^T e^{-\delta t} [p q E B*(E) - c E] dt
               = [p q E B*(E) - c E] * (1 - e^{-\delta T}) / \delta

    Parameters
    ----------
    E : float
        捕捞努力量
    r, K, q : float
        种群动态参数
    p : float
        单位鱼价
    c : float
        单位捕捞成本
    delta : float
        贴现率
    T : float
        时间跨度

    Returns
    -------
    neg_profit : float
        负利润（用于最小化）
    """
    # HOLE 1: Implement the discounted profit objective function
    # 核心科学计算：
    # 1. 计算稳态生物量 B*(E) = schaefer_gordon_steady_state(E, r, K, q)
    # 2. 计算单位时间利润 profit_rate = p * q * E * B - c * E
    # 3. 计算贴现因子 discount_factor = (1 - exp(-delta * T)) / delta
    # 4. 返回负总利润 -profit_rate * discount_factor
    pass


def find_optimal_effort(r, K, q, p, c, delta, T=50.0, E_max=None):
    """
    利用 Brent 优化求解最优捕捞努力量 E*

    约束条件：0 <= E <= E_max
    若 E_max 为 None，则取 E_max = r/q（生物学最大可持续努力量）

    Parameters
    ----------
    r, K, q, p, c, delta : float
        模型参数
    T : float
        时间跨度
    E_max : float, optional
        最大允许努力量

    Returns
    -------
    E_opt : float
        最优捕捞努力量
    profit_opt : float
        对应的最大贴现利润
    """
    if E_max is None:
        E_max = r / q

    def obj(E):
        return discounted_profit_objective(E, r, K, q, p, c, delta, T)

    optimizer = BrentOptimizer(0.0, E_max)
    E_opt, neg_profit = optimizer.optimize(obj)
    return E_opt, -neg_profit


def bellman_ford_shortest_paths(v_num, e_list, e_weight, source):
    """
    Bellman-Ford 算法求解带权有向图单源最短路径

    在渔业 MPA 网络优化中的应用：
    - 节点：各海洋保护区或渔区
    - 边：生态连通性通道或资源调配路线
    - 边权重：距离、生态阻力或经济成本（可为负，表示补贴）

    算法步骤：
    1. 初始化：dist[source] = 0，其余为 INF
    2. 松弛：重复 V-1 次，对所有边 (u,v) 执行
       若 dist[u] + w(u,v) < dist[v]，则更新 dist[v] 和 predecessor[v]
    3. 负环检测：若还能松弛，则存在负权环

    Parameters
    ----------
    v_num : int
        顶点数
    e_list : list of tuple
        边列表，每个元素为 (u, v) 的顶点对（0-based索引）
    e_weight : list or ndarray
        边权重数组，与 e_list 一一对应
    source : int
        源顶点索引

    Returns
    -------
    dist : ndarray
        从源点到各顶点的最短距离
    predecessor : ndarray
        前驱节点数组，用于路径重构
    """
    e_weight = np.asarray(e_weight, dtype=float)
    dist = np.full(v_num, NumericalConfig.R8_BIG, dtype=float)
    dist[source] = 0.0
    predecessor = np.full(v_num, -1, dtype=int)

    # 松弛 V-1 次
    for _ in range(v_num - 1):
        updated = False
        for j, (u, v) in enumerate(e_list):
            if dist[u] + e_weight[j] < dist[v] - NumericalConfig.TOL:
                dist[v] = dist[u] + e_weight[j]
                predecessor[v] = u
                updated = True
        if not updated:
            break

    # 负环检测
    for j, (u, v) in enumerate(e_list):
        if dist[u] + e_weight[j] < dist[v] - NumericalConfig.TOL:
            raise RuntimeError("Graph contains a negative-weight cycle")

    return dist, predecessor


def mpa_network_optimize(n_patches, connectivity_matrix, source_patch=0):
    """
    海洋保护区网络连通性优化

    将连通性矩阵转换为边列表，应用 Bellman-Ford 算法
    计算从源保护区到所有其他保护区的最优生态路径

    Parameters
    ----------
    n_patches : int
        保护区数量
    connectivity_matrix : ndarray, shape (n_patches, n_patches)
        连通性矩阵，entry[i,j] 表示从 i 到 j 的连通性成本
        若 entry[i,j] = inf 表示无直接连接
    source_patch : int
        源保护区索引

    Returns
    -------
    dist : ndarray
        最短生态路径成本
    predecessor : ndarray
        前驱节点
    """
    e_list = []
    e_weight = []
    for i in range(n_patches):
        for j in range(n_patches):
            if i != j and not np.isinf(connectivity_matrix[i, j]):
                e_list.append((i, j))
                e_weight.append(connectivity_matrix[i, j])

    dist, predecessor = bellman_ford_shortest_paths(n_patches, e_list, e_weight, source_patch)
    return dist, predecessor


def reconstruct_path(predecessor, target):
    """
    根据前驱数组重构从源点到目标点的最短路径

    Parameters
    ----------
    predecessor : ndarray
        Bellman-Ford 返回的前驱数组
    target : int
        目标顶点

    Returns
    -------
    path : list
        顶点索引列表，从源点到目标点
    """
    path = []
    v = target
    while v != -1:
        path.append(int(v))
        v = predecessor[v]
    path.reverse()
    return path
