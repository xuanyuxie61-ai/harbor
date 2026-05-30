
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum


class MarketState(Enum):
    OPEN_AUCTION = 0
    CONTINUOUS = 1
    HIGH_VOLATILITY = 2
    CLOSE_AUCTION = 3


class OrderType(Enum):
    MARKET_BUY = 0
    MARKET_SELL = 1
    LIMIT_BUY = 2
    LIMIT_SELL = 3
    CANCEL = 4


class MarketEvent:

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

    def __init__(self,
                 state_durations: Dict[MarketState, Tuple[float, float]],
                 transition_probs: Optional[np.ndarray] = None,
                 initial_state: MarketState = MarketState.CONTINUOUS):
        self.state = initial_state
        self.state_durations = state_durations
        self.timer = 0.0
        self.cycle_count = 0

        if transition_probs is None:

            self.transition_probs = np.array([
                [0.1, 0.7, 0.15, 0.05],
                [0.05, 0.8, 0.12, 0.03],
                [0.05, 0.65, 0.25, 0.05],
                [0.3, 0.4, 0.2, 0.1],
            ])
        else:
            self.transition_probs = transition_probs

        self._validate_transition_matrix()
        self._draw_next_duration()

    def _validate_transition_matrix(self):
        if self.transition_probs.shape != (4, 4):
            raise ValueError("转移矩阵必须是 4x4.")
        if not np.allclose(np.sum(self.transition_probs, axis=1), 1.0):
            raise ValueError("转移矩阵每行必须和为 1.")
        if np.any(self.transition_probs < 0):
            raise ValueError("转移矩阵元素必须非负.")

    def _draw_next_duration(self):
        mean, jitter = self.state_durations.get(
            self.state, (30.0, 3.0)
        )

        duration = np.random.normal(mean, jitter)
        self.target_duration = max(duration, 1.0)
        self.timer = 0.0

    def step(self, dt: float) -> bool:
        self.timer += dt
        if self.timer >= self.target_duration:
            self._transition()
            return True
        return False

    def _transition(self):
        current_idx = self.state.value
        probs = self.transition_probs[current_idx]
        next_idx = np.random.choice(4, p=probs)
        self.state = MarketState(next_idx)
        self.cycle_count += 1
        self._draw_next_duration()

    def get_intensity_multipliers(self) -> Dict[OrderType, float]:
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

    def __init__(self,
                 base_intensities: Dict[OrderType, float],
                 price_range: Tuple[float, float] = (99.0, 101.0),
                 volume_dist: str = "geometric"):
        self.base_intensities = base_intensities
        self.price_range = price_range
        self.volume_dist = volume_dist

    def generate_events(self, t_start: float, t_end: float,
                        state_machine: MarketStateMachine,
                        seed: Optional[int] = None) -> List[MarketEvent]:
        if seed is not None:
            np.random.seed(seed)

        events: List[MarketEvent] = []
        current_time = t_start
        sm = state_machine


        dt_slice = 0.001

        while current_time < t_end:
            multipliers = sm.get_intensity_multipliers()
            for otype, base_lambda in self.base_intensities.items():
                lam = base_lambda * multipliers.get(otype, 1.0)
                if lam <= 0.0:
                    continue

                expected = lam * dt_slice

                n_arrivals = np.random.poisson(expected)
                for _ in range(n_arrivals):

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


            sm.step(dt_slice)
            current_time += dt_slice

        events.sort(key=lambda e: e.time)
        return events

    def _draw_price(self, otype: OrderType) -> Optional[float]:
        if otype in (OrderType.MARKET_BUY, OrderType.MARKET_SELL):
            return None
        return np.random.uniform(self.price_range[0], self.price_range[1])

    def _draw_volume(self) -> int:
        if self.volume_dist == "geometric":

            p = 0.5
            vol = np.random.geometric(p)
            return min(vol, 100)
        else:
            vol = np.random.poisson(5) + 1
            return min(vol, 100)


class MarketSimulator:

    def __init__(self,
                 duration_seconds: float = 60.0,
                 base_price: float = 100.0,
                 seed: Optional[int] = None):
        self.duration = duration_seconds
        self.base_price = base_price
        self.seed = seed


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
