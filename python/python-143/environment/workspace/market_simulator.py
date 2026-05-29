"""
market_simulator.py
===================
高频市场微观结构离散事件模拟引擎

本模块基于以下种子项目融合:
- 1284_traffic_simulation: 交通流排队模拟 → 订单流泊松到达+队列状态机
- 1164_stiff_ode: 刚性ODE参数管理 → 市场状态转换的持久化参数

核心数学模型:
--------------
1.  泊松订单到达过程:
    在时间区间 [t, t+Δt] 内, 类型为 k 的订单到达数量 N_k 服从:
        P(N_k = n) = (λ_k Δt)^n / n! * exp(-λ_k Δt)
    其中 λ_k 为到达强度 (orders/second).

    对高频交易 (HFT), 典型参数:
        λ_market ≈ 100~500 笔/秒 (市价单)
        λ_limit  ≈ 1000~5000 笔/秒 (限价单)
        λ_cancel ≈ 500~2000 笔/秒 (撤单)

2.  市场状态机 (Market Regime State Machine):
    市场存在离散状态 M_t ∈ {开盘集合竞价, 连续交易, 波动聚集, 收盘}.
    状态转移满足连续时间马尔可夫链 (CTMC):
        dP(M_t = j | M_t = i) = q_{ij} dt
    其中 q_{ij} 为转移速率矩阵的 (i,j) 元素.

3.  排队延迟与执行概率:
    订单在LOB中的等待时间 W 服从服务时间分布:
        对市价单: 服务率 μ_market = C / Q
        其中 C 为撮合通道容量 (orders/sec), Q 为队列长度.
    排队论中的 Little 定律:
        L = λ W
        其中 L 为平均队列长度, λ 为到达率, W 为平均等待时间.

4.  事件驱动模拟框架:
    采用离散事件模拟 (DES) 而非固定时间步长,
    下一个事件时间:
        τ_next = min_k { τ_k },   τ_k ~ Exp(λ_k)
    其中 Exp(λ_k) 为指数分布随机变量.

5.  成交量加权平均价格 (VWAP) 基准:
        VWAP = Σ (P_i * V_i) / Σ V_i
    策略执行质量的衡量标准.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum


class MarketState(Enum):
    """市场离散状态."""
    OPEN_AUCTION = 0
    CONTINUOUS = 1
    HIGH_VOLATILITY = 2
    CLOSE_AUCTION = 3


class OrderType(Enum):
    """订单类型."""
    MARKET_BUY = 0
    MARKET_SELL = 1
    LIMIT_BUY = 2
    LIMIT_SELL = 3
    CANCEL = 4


class MarketEvent:
    """离散事件记录."""

    def __init__(self, time: float, order_type: OrderType,
                 price: Optional[float] = None,
                 volume: int = 1,
                 state: MarketState = MarketState.CONTINUOUS):
        self.time = time
        self.order_type = order_type
        self.price = price
        self.volume = volume
        self.state = state

    def __repr__(self) -> str:
        return (f"MarketEvent(t={self.time:.6f}, type={self.order_type.name}, "
                f"price={self.price}, vol={self.volume}, state={self.state.name})")


class MarketStateMachine:
    """
    市场状态机, 基于 1284_traffic_simulation 中红绿灯状态切换的思想.
    市场在不同 regime 之间切换, 影响订单到达率.
    """

    def __init__(self,
                 state_durations: Dict[MarketState, Tuple[float, float]],
                 transition_probs: Optional[np.ndarray] = None,
                 initial_state: MarketState = MarketState.CONTINUOUS):
        """
        Parameters
        ----------
        state_durations : dict
            每个状态的平均持续时长 (秒) 及其抖动范围.
            例如: {MarketState.CONTINUOUS: (60.0, 5.0)}
        transition_probs : np.ndarray, shape (4,4)
            状态转移概率矩阵 (离散时间近似).
        initial_state : MarketState
            初始状态.
        """
        self.state = initial_state
        self.state_durations = state_durations
        self.timer = 0.0
        self.cycle_count = 0

        if transition_probs is None:
            # 默认: 主要在连续交易和波动聚集之间切换
            self.transition_probs = np.array([
                [0.1, 0.7, 0.15, 0.05],   # OPEN_AUCTION
                [0.05, 0.8, 0.12, 0.03],  # CONTINUOUS
                [0.05, 0.65, 0.25, 0.05], # HIGH_VOLATILITY
                [0.3, 0.4, 0.2, 0.1],     # CLOSE_AUCTION
            ])
        else:
            self.transition_probs = transition_probs

        self._validate_transition_matrix()
        self._draw_next_duration()

    def _validate_transition_matrix(self):
        """验证转移概率矩阵的合法性."""
        if self.transition_probs.shape != (4, 4):
            raise ValueError("转移矩阵必须是 4x4.")
        if not np.allclose(np.sum(self.transition_probs, axis=1), 1.0):
            raise ValueError("转移矩阵每行必须和为 1.")
        if np.any(self.transition_probs < 0):
            raise ValueError("转移矩阵元素必须非负.")

    def _draw_next_duration(self):
        """抽取下一个状态的持续时长."""
        mean, jitter = self.state_durations.get(
            self.state, (30.0, 3.0)
        )
        # 采用截断正态分布, 避免负值
        duration = np.random.normal(mean, jitter)
        self.target_duration = max(duration, 1.0)
        self.timer = 0.0

    def step(self, dt: float) -> bool:
        """
        推进一个时间步长, 检查是否需要状态切换.

        Returns
        -------
        changed : bool
            若发生状态切换返回 True.
        """
        self.timer += dt
        if self.timer >= self.target_duration:
            self._transition()
            return True
        return False

    def _transition(self):
        """执行状态转移."""
        current_idx = self.state.value
        probs = self.transition_probs[current_idx]
        next_idx = np.random.choice(4, p=probs)
        self.state = MarketState(next_idx)
        self.cycle_count += 1
        self._draw_next_duration()

    def get_intensity_multipliers(self) -> Dict[OrderType, float]:
        """
        根据当前市场状态返回各订单类型的强度乘数.
        不同状态下订单到达率不同:
            波动聚集期: 市价单和撤单激增
            集合竞价期: 限价单占主导
        """
        multipliers = {
            MarketState.OPEN_AUCTION: {
                OrderType.MARKET_BUY: 0.2,
                OrderType.MARKET_SELL: 0.2,
                OrderType.LIMIT_BUY: 2.5,
                OrderType.LIMIT_SELL: 2.5,
                OrderType.CANCEL: 0.5,
            },
            MarketState.CONTINUOUS: {
                OrderType.MARKET_BUY: 1.0,
                OrderType.MARKET_SELL: 1.0,
                OrderType.LIMIT_BUY: 1.0,
                OrderType.LIMIT_SELL: 1.0,
                OrderType.CANCEL: 1.0,
            },
            MarketState.HIGH_VOLATILITY: {
                OrderType.MARKET_BUY: 3.0,
                OrderType.MARKET_SELL: 3.0,
                OrderType.LIMIT_BUY: 0.8,
                OrderType.LIMIT_SELL: 0.8,
                OrderType.CANCEL: 2.5,
            },
            MarketState.CLOSE_AUCTION: {
                OrderType.MARKET_BUY: 0.3,
                OrderType.MARKET_SELL: 0.3,
                OrderType.LIMIT_BUY: 2.0,
                OrderType.LIMIT_SELL: 2.0,
                OrderType.CANCEL: 0.8,
            },
        }
        return multipliers.get(self.state, multipliers[MarketState.CONTINUOUS])


class OrderFlowGenerator:
    """
    订单流生成器, 基于泊松过程.
    结合交通流模拟中车辆到达的排队思想.
    """

    def __init__(self,
                 base_intensities: Dict[OrderType, float],
                 price_range: Tuple[float, float] = (99.0, 101.0),
                 volume_dist: str = "geometric"):
        """
        Parameters
        ----------
        base_intensities : dict
            各订单类型的基准到达强度 (orders/sec).
        price_range : tuple
            限价单价格的均匀分布范围.
        volume_dist : str
            成交量分布: "geometric" 或 "poisson".
        """
        self.base_intensities = base_intensities
        self.price_range = price_range
        self.volume_dist = volume_dist

    def generate_events(self, t_start: float, t_end: float,
                        state_machine: MarketStateMachine,
                        seed: Optional[int] = None) -> List[MarketEvent]:
        """
        在区间 [t_start, t_end] 内生成离散事件序列.

        算法:
        1.  对每个订单类型 k, 生成泊松过程事件时间:
                τ_i^{(k)} = Σ_{j=1}^i E_j^{(k)} / λ_k
            其中 E_j^{(k)} ~ Exp(1) i.i.d.
        2.  合并所有类型的事件, 按时间排序.
        3.  在每个事件点更新市场状态机.

        Returns
        -------
        events : List[MarketEvent]
            按时间排序的事件列表.
        """
        if seed is not None:
            np.random.seed(seed)

        events: List[MarketEvent] = []
        current_time = t_start
        sm = state_machine

        # 为了效率, 采用逐个小时间片生成
        dt_slice = 0.001  # 1ms 时间片

        while current_time < t_end:
            multipliers = sm.get_intensity_multipliers()
            for otype, base_lambda in self.base_intensities.items():
                lam = base_lambda * multipliers.get(otype, 1.0)
                if lam <= 0.0:
                    continue
                # 在 dt_slice 内期望到达数量
                expected = lam * dt_slice
                # 泊松抽样
                n_arrivals = np.random.poisson(expected)
                for _ in range(n_arrivals):
                    # 事件发生在时间片内的均匀随机位置
                    t_event = current_time + np.random.uniform(0.0, dt_slice)
                    price = self._draw_price(otype)
                    volume = self._draw_volume()
                    events.append(MarketEvent(
                        time=t_event,
                        order_type=otype,
                        price=price,
                        volume=volume,
                        state=sm.state
                    ))

            # 推进状态机
            sm.step(dt_slice)
            current_time += dt_slice

        events.sort(key=lambda e: e.time)
        return events

    def _draw_price(self, otype: OrderType) -> Optional[float]:
        """抽取订单价格."""
        if otype in (OrderType.MARKET_BUY, OrderType.MARKET_SELL):
            return None
        return np.random.uniform(self.price_range[0], self.price_range[1])

    def _draw_volume(self) -> int:
        """抽取订单成交量."""
        if self.volume_dist == "geometric":
            # 几何分布: P(V=k) = (1-p)^{k-1} p, k≥1
            p = 0.5
            vol = np.random.geometric(p)
            return min(vol, 100)
        else:
            vol = np.random.poisson(5) + 1
            return min(vol, 100)


class MarketSimulator:
    """
    高频市场模拟器主类.
    整合状态机与订单流生成器, 提供统一接口.
    """

    def __init__(self,
                 duration_seconds: float = 60.0,
                 base_price: float = 100.0,
                 seed: Optional[int] = None):
        self.duration = duration_seconds
        self.base_price = base_price
        self.seed = seed

        # 默认参数配置 (基于真实高频市场特征)
        self.base_intensities = {
            OrderType.MARKET_BUY: 50.0,
            OrderType.MARKET_SELL: 50.0,
            OrderType.LIMIT_BUY: 200.0,
            OrderType.LIMIT_SELL: 200.0,
            OrderType.CANCEL: 150.0,
        }

        self.state_durations = {
            MarketState.OPEN_AUCTION: (5.0, 1.0),
            MarketState.CONTINUOUS: (30.0, 5.0),
            MarketState.HIGH_VOLATILITY: (10.0, 2.0),
            MarketState.CLOSE_AUCTION: (5.0, 1.0),
        }

    def run(self) -> Tuple[List[MarketEvent], MarketStateMachine]:
        """
        执行一次市场模拟.

        Returns
        -------
        events : List[MarketEvent]
            模拟产生的所有市场事件.
        sm : MarketStateMachine
            最终的市场状态机 (含统计信息).
        """
        sm = MarketStateMachine(
            state_durations=self.state_durations,
            initial_state=MarketState.CONTINUOUS
        )
        generator = OrderFlowGenerator(
            base_intensities=self.base_intensities,
            price_range=(self.base_price - 1.0, self.base_price + 1.0)
        )
        events = generator.generate_events(
            t_start=0.0,
            t_end=self.duration,
            state_machine=sm,
            seed=self.seed
        )
        return events, sm

    @staticmethod
    def compute_vwap(events: List[MarketEvent]) -> Optional[float]:
        """
        计算成交量加权平均价格 VWAP.
            VWAP = Σ (P_i * V_i) / Σ V_i
        仅对带价格的成交事件有效.
        """
        total_pv = 0.0
        total_v = 0
        for e in events:
            if e.price is not None and e.volume > 0:
                total_pv += e.price * e.volume
                total_v += e.volume
        if total_v == 0:
            return None
        return total_pv / total_v

    @staticmethod
    def compute_order_imbalance(events: List[MarketEvent]) -> float:
        """
        计算买卖订单流不平衡度 (Order Imbalance):
            Imb = (V_buy - V_sell) / (V_buy + V_sell)
        取值范围 [-1, 1], 接近 0 表示均衡.
        """
        v_buy = 0
        v_sell = 0
        for e in events:
            if e.order_type in (OrderType.MARKET_BUY, OrderType.LIMIT_BUY):
                v_buy += e.volume
            elif e.order_type in (OrderType.MARKET_SELL, OrderType.LIMIT_SELL):
                v_sell += e.volume
        total = v_buy + v_sell
        if total == 0:
            return 0.0
        return (v_buy - v_sell) / total
