
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PriceLevel:
    price: float
    bid_volume: int
    ask_volume: int


class LimitOrderBook:

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


        self.levels: Dict[int, PriceLevel] = {}
        for i in range(-self.half_levels, self.half_levels + 1):
            p = base_price + i * tick_size
            self.levels[i] = PriceLevel(price=p, bid_volume=0, ask_volume=0)


        self.total_bid_volume = 0
        self.total_ask_volume = 0
        self.trade_history: List[Tuple[float, int, float]] = []

    def _price_to_index(self, price: float) -> int:
        idx = int(np.round((price - self.base_price) / self.tick_size))
        return idx

    def add_order(self, price: float, volume: int, is_bid: bool):
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
        remaining = volume
        total_cost = 0.0

        if is_buy:

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
        bb, ba, _, _ = self.best_quotes()
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)

    def spread(self) -> Optional[float]:
        bb, ba, _, _ = self.best_quotes()
        if bb is None or ba is None:
            return None
        return ba - bb

    def depth_profile(self, side: str = "bid", n_levels: int = 10) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def quadratic_interpolate(x_nodes: np.ndarray,
                               y_nodes: np.ndarray,
                               x_query: np.ndarray) -> np.ndarray:
        if len(x_nodes) < 3:

            return np.interp(x_query, x_nodes, y_nodes)

        y_query = np.zeros_like(x_query)
        for k, xq in enumerate(x_query):

            idx = np.searchsorted(x_nodes, xq)
            if idx == 0:
                idx = 1
            if idx >= len(x_nodes) - 1:
                idx = len(x_nodes) - 2


            i0 = max(0, idx - 1)
            i1 = idx
            i2 = min(len(x_nodes) - 1, idx + 1)

            x0, x1, x2 = x_nodes[i0], x_nodes[i1], x_nodes[i2]
            y0, y1, y2 = y_nodes[i0], y_nodes[i1], y_nodes[i2]


            L0 = ((xq - x1) * (xq - x2)) / ((x0 - x1) * (x0 - x2) + 1e-18)
            L1 = ((xq - x0) * (xq - x2)) / ((x1 - x0) * (x1 - x2) + 1e-18)
            L2 = ((xq - x0) * (xq - x1)) / ((x2 - x0) * (x2 - x1) + 1e-18)

            y_query[k] = y0 * L0 + y1 * L1 + y2 * L2

        return y_query


class LOBGeometryAnalyzer:

    @staticmethod
    def quadrilateral_area(quad: np.ndarray) -> float:
        if quad.shape != (2, 4):
            raise ValueError("quad 必须是 2x4 数组.")


        t1 = quad[:, [0, 1, 2]]
        area1 = 0.5 * abs(
            (t1[0, 1] - t1[0, 0]) * (t1[1, 2] - t1[1, 0])
            - (t1[0, 2] - t1[0, 0]) * (t1[1, 1] - t1[1, 0])
        )


        t2 = quad[:, [0, 2, 3]]
        area2 = 0.5 * abs(
            (t2[0, 1] - t2[0, 0]) * (t2[1, 2] - t2[1, 0])
            - (t2[0, 2] - t2[0, 0]) * (t2[1, 1] - t2[1, 0])
        )

        return area1 + area2

    @staticmethod
    def lob_convexity(prices: np.ndarray, depths: np.ndarray) -> float:
        if len(prices) < 3:
            return 0.0

        second_diff = np.diff(depths, n=2)
        positive_ratio = np.sum(second_diff > 0) / len(second_diff)
        return positive_ratio

    @staticmethod
    def depth_concentration(depths: np.ndarray) -> float:
        total = np.sum(depths)
        if total <= 0:
            return 0.0
        shares = depths / total
        return np.sum(shares ** 2)

    @staticmethod
    def market_thickness(spread: float, max_depth: float) -> float:
        if spread <= 0 or max_depth <= 0:
            return 0.0
        return spread * max_depth
