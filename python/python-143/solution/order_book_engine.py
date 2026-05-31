"""
order_book_engine.py
====================
限价订单簿 (LOB) 引擎与高阶深度曲面插值

本模块基于以下种子项目融合:
- 492_gridlines: 网格生成 → 价格-时间离散网格上的订单簿切片
- 1233_tet_mesh_l2q: 线性到二次网格升阶 → LOB深度曲面从线性插值到二次插值
- 952_quadrilateral: 四边形几何计算 → 订单簿四边形区域的面积与凸性分析

核心数学模型:
--------------
1.  限价订单簿离散表示:
    设价格层级为 p_i = p_0 + i * tick_size, i ∈ ℤ.
    在每个层级上, 买单深度 D_bid(p_i) ≥ 0, 卖单深度 D_ask(p_i) ≥ 0.
    订单簿状态向量:
        L(t) = { (p_i, D_bid(p_i), D_ask(p_i)) }_{i=-N}^{N}

2.  最优报价 (Best Quotes):
        P_bid(t) = max { p_i : D_bid(p_i) > 0 }
        P_ask(t) = min { p_i : D_ask(p_i) > 0 }
        买卖价差:  S(t) = P_ask(t) - P_bid(t) ≥ tick_size

3.  中间价与加权中间价:
        P_mid(t) = (P_ask + P_bid) / 2
        P_wmid(t) = (P_ask * Q_bid + P_bid * Q_ask) / (Q_bid + Q_ask)
    其中 Q_bid, Q_ask 为最优报价处的深度.

4.  LOB 深度曲面插值 (基于 tet_mesh_l2q 升阶思想):
    给定离散深度值 D_i = D(p_i), 构造二次样条插值:
        D̃(p) = Σ_{j} c_j φ_j(p)
    其中 φ_j 为二次 Lagrange 基函数:
        φ_j(p) = Π_{k≠j} (p - p_k) / (p_j - p_k)
    高阶插值可更准确估计大额订单冲击成本.

5.  四边形价格-深度区域分析 (基于 quadrilateral 思想):
    在 (价格, 深度) 平面上, 订单簿的四边形包围盒:
        Q = [P_bid, P_ask] × [0, D_max]
    其面积 A(Q) = S(t) * D_max(t) 可视为市场"厚度"的度量.
    凸性检测: 若深度函数在 [P_bid, P_ask] 上凸, 则大单冲击成本递增.

6.  冲击成本模型 (Market Impact):
    对买入量 Q 的瞬时价格冲击:
        ΔP(Q) = γ * Q^δ
    其中 γ > 0 为冲击系数, δ ≈ 0.5~1.0 (平方根定律).
    利用 LOB 深度曲面插值, 可数值计算:
        ΔP(Q) = argmin_p { ∫_{P_ask}^{p} D̃_ask(x) dx ≥ Q } - P_ask
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PriceLevel:
    """单个价格层级的数据结构."""
    price: float
    bid_volume: int
    ask_volume: int


class LimitOrderBook:
    """
    限价订单簿核心引擎.
    """

    def __init__(self, base_price: float = 100.0,
                 tick_size: float = 0.01,
                 max_levels: int = 200):
        if tick_size <= 0.0:
            raise ValueError("tick_size 必须为正.")
        if max_levels <= 0:
            raise ValueError("max_levels 必须为正整数.")

        self.base_price = base_price
        self.tick_size = tick_size
        self.max_levels = max_levels
        self.half_levels = max_levels // 2

        # 初始化对称的价格层级
        self.levels: Dict[int, PriceLevel] = {}
        for i in range(-self.half_levels, self.half_levels + 1):
            p = base_price + i * tick_size
            self.levels[i] = PriceLevel(price=p, bid_volume=0, ask_volume=0)

        # 统计量
        self.total_bid_volume = 0
        self.total_ask_volume = 0
        self.trade_history: List[Tuple[float, int, float]] = []  # (time, vol, price)

    def _price_to_index(self, price: float) -> int:
        """将价格映射到离散层级索引."""
        idx = int(np.round((price - self.base_price) / self.tick_size))
        return idx

    def add_order(self, price: float, volume: int, is_bid: bool):
        """添加限价订单."""
        idx = self._price_to_index(price)
        if idx not in self.levels:
            return
        level = self.levels[idx]
        if is_bid:
            level.bid_volume += volume
            self.total_bid_volume += volume
        else:
            level.ask_volume += volume
            self.total_ask_volume += volume

    def cancel_order(self, price: float, volume: int, is_bid: bool):
        """撤单."""
        idx = self._price_to_index(price)
        if idx not in self.levels:
            return
        level = self.levels[idx]
        if is_bid:
            cancel_vol = min(volume, level.bid_volume)
            level.bid_volume -= cancel_vol
            self.total_bid_volume -= cancel_vol
        else:
            cancel_vol = min(volume, level.ask_volume)
            level.ask_volume -= cancel_vol
            self.total_ask_volume -= cancel_vol

    def execute_market_order(self, volume: int, is_buy: bool,
                             current_time: float = 0.0) -> Tuple[int, float]:
        """
        执行市价单, 返回实际成交量与成交均价.

        对买入市价单: 从最优卖价开始逐个层级吃掉 ask 深度.
        对卖出市价单: 从最优买价开始逐个层级吃掉 bid 深度.
        """
        remaining = volume
        total_cost = 0.0

        if is_buy:
            # 从最低卖价向上遍历
            indices = sorted(self.levels.keys())
            for idx in indices:
                level = self.levels[idx]
                if level.ask_volume <= 0:
                    continue
                exec_vol = min(remaining, level.ask_volume)
                total_cost += exec_vol * level.price
                level.ask_volume -= exec_vol
                self.total_ask_volume -= exec_vol
                remaining -= exec_vol
                if exec_vol > 0:
                    self.trade_history.append(
                        (current_time, exec_vol, level.price)
                    )
                if remaining <= 0:
                    break
        else:
            # 从最高买价向下遍历
            indices = sorted(self.levels.keys(), reverse=True)
            for idx in indices:
                level = self.levels[idx]
                if level.bid_volume <= 0:
                    continue
                exec_vol = min(remaining, level.bid_volume)
                total_cost += exec_vol * level.price
                level.bid_volume -= exec_vol
                self.total_bid_volume -= exec_vol
                remaining -= exec_vol
                if exec_vol > 0:
                    self.trade_history.append(
                        (current_time, exec_vol, level.price)
                    )
                if remaining <= 0:
                    break

        executed = volume - remaining
        avg_price = total_cost / executed if executed > 0 else 0.0
        return executed, avg_price

    def best_quotes(self) -> Tuple[Optional[float], Optional[float],
                                    Optional[int], Optional[int]]:
        """
        返回最优买卖报价及对应深度.
        """
        best_bid = None
        best_bid_vol = None
        best_ask = None
        best_ask_vol = None

        for idx in sorted(self.levels.keys(), reverse=True):
            level = self.levels[idx]
            if best_bid is None and level.bid_volume > 0:
                best_bid = level.price
                best_bid_vol = level.bid_volume
            if best_ask is None and level.ask_volume > 0:
                best_ask = level.price
                best_ask_vol = level.ask_volume
            if best_bid is not None and best_ask is not None:
                break

        return best_bid, best_ask, best_bid_vol, best_ask_vol

    def mid_price(self) -> Optional[float]:
        """中间价."""
        bb, ba, _, _ = self.best_quotes()
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)

    def spread(self) -> Optional[float]:
        """买卖价差."""
        bb, ba, _, _ = self.best_quotes()
        if bb is None or ba is None:
            return None
        return ba - bb

    def depth_profile(self, side: str = "bid", n_levels: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取某侧的n个最优层级的深度剖面.

        Returns
        -------
        prices : np.ndarray
        depths : np.ndarray
        """
        prices = []
        depths = []

        if side == "bid":
            for idx in sorted(self.levels.keys(), reverse=True):
                level = self.levels[idx]
                if level.bid_volume > 0:
                    prices.append(level.price)
                    depths.append(level.bid_volume)
                if len(prices) >= n_levels:
                    break
        else:
            for idx in sorted(self.levels.keys()):
                level = self.levels[idx]
                if level.ask_volume > 0:
                    prices.append(level.price)
                    depths.append(level.ask_volume)
                if len(prices) >= n_levels:
                    break

        return np.array(prices), np.array(depths)


class LOBInterpolator:
    """
    LOB 深度曲面的高阶插值, 基于 1233_tet_mesh_l2q 的升阶思想.
    将离散深度数据从线性插值提升到二次多项式插值.
    """

    @staticmethod
    def quadratic_interpolate(x_nodes: np.ndarray,
                               y_nodes: np.ndarray,
                               x_query: np.ndarray) -> np.ndarray:
        """
        二次 Lagrange 插值.
        对每三个相邻节点 (x_i, x_{i+1}, x_{i+2}) 构造二次多项式.

        基函数:
            L_0(x) = (x-x_1)(x-x_2) / ((x_0-x_1)(x_0-x_2))
            L_1(x) = (x-x_0)(x-x_2) / ((x_1-x_0)(x_1-x_2))
            L_2(x) = (x-x_0)(x-x_1) / ((x_2-x_0)(x_2-x_1))

        插值多项式:
            P(x) = y_0 L_0(x) + y_1 L_1(x) + y_2 L_2(x)
        """
        if len(x_nodes) < 3:
            # 退化为线性插值
            return np.interp(x_query, x_nodes, y_nodes)

        y_query = np.zeros_like(x_query)
        for k, xq in enumerate(x_query):
            # 找到包含 xq 的区间
            idx = np.searchsorted(x_nodes, xq)
            if idx == 0:
                idx = 1
            if idx >= len(x_nodes) - 1:
                idx = len(x_nodes) - 2

            # 取左中右三个节点
            i0 = max(0, idx - 1)
            i1 = idx
            i2 = min(len(x_nodes) - 1, idx + 1)

            x0, x1, x2 = x_nodes[i0], x_nodes[i1], x_nodes[i2]
            y0, y1, y2 = y_nodes[i0], y_nodes[i1], y_nodes[i2]

            # Lagrange 基函数
            L0 = ((xq - x1) * (xq - x2)) / ((x0 - x1) * (x0 - x2) + 1e-18)
            L1 = ((xq - x0) * (xq - x2)) / ((x1 - x0) * (x1 - x2) + 1e-18)
            L2 = ((xq - x0) * (xq - x1)) / ((x2 - x0) * (x2 - x1) + 1e-18)

            y_query[k] = y0 * L0 + y1 * L1 + y2 * L2

        return y_query


class LOBGeometryAnalyzer:
    """
    订单簿几何分析器, 基于 952_quadrilateral 的思想.
    将 LOB 的买卖深度分布视为几何对象, 分析其面积、凸性和集中度.
    """

    @staticmethod
    def quadrilateral_area(quad: np.ndarray) -> float:
        """
        计算四边形面积 (可处理非凸四边形).
        将四边形沿对角线分成两个三角形求和.

        三角形面积 (叉积公式):
            A = 0.5 | (x_2-x_1)(y_3-y_1) - (x_3-x_1)(y_2-y_1) |
        """
        if quad.shape != (2, 4):
            raise ValueError("quad 必须是 2x4 数组.")

        # 三角形 1: 顶点 0,1,2
        t1 = quad[:, [0, 1, 2]]
        area1 = 0.5 * abs(
            (t1[0, 1] - t1[0, 0]) * (t1[1, 2] - t1[1, 0])
            - (t1[0, 2] - t1[0, 0]) * (t1[1, 1] - t1[1, 0])
        )

        # 三角形 2: 顶点 0,2,3
        t2 = quad[:, [0, 2, 3]]
        area2 = 0.5 * abs(
            (t2[0, 1] - t2[0, 0]) * (t2[1, 2] - t2[1, 0])
            - (t2[0, 2] - t2[0, 0]) * (t2[1, 1] - t2[1, 0])
        )

        return area1 + area2

    @staticmethod
    def lob_convexity(prices: np.ndarray, depths: np.ndarray) -> float:
        """
        检测深度剖面的凸性.
        对离散数据, 计算二阶差分的符号:
            Δ²D_i = D_{i+1} - 2D_i + D_{i-1}
        若大部分 Δ²D_i > 0, 则深度函数呈凸性.

        返回凸性比率: 正二阶差分占比.
        """
        if len(prices) < 3:
            return 0.0

        second_diff = np.diff(depths, n=2)
        positive_ratio = np.sum(second_diff > 0) / len(second_diff)
        return positive_ratio

    @staticmethod
    def depth_concentration(depths: np.ndarray) -> float:
        """
        深度集中度 (赫芬达尔指数变体):
            H = Σ (d_i / D_total)²
        H ∈ [1/n, 1], 越大表示深度越集中在少数层级.
        """
        total = np.sum(depths)
        if total <= 0:
            return 0.0
        shares = depths / total
        return np.sum(shares ** 2)

    @staticmethod
    def market_thickness(spread: float, max_depth: float) -> float:
        """
        市场"厚度"指标: 价差与最大深度的乘积.
        类比四边形面积.
        """
        if spread <= 0 or max_depth <= 0:
            return 0.0
        return spread * max_depth
