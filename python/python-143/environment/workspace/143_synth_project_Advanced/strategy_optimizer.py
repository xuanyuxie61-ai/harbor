
import numpy as np
from typing import Callable, Tuple, List, Optional


class MarketMakingStrategy:

    def __init__(self,
                 delta_bid: float = 0.02,
                 delta_ask: float = 0.02,
                 inventory_target: int = 0,
                 inventory_penalty: float = 0.001,
                 max_inventory: int = 100,
                 order_size: int = 1):
        self.delta_bid = delta_bid
        self.delta_ask = delta_ask
        self.inventory_target = inventory_target
        self.inventory_penalty = inventory_penalty
        self.max_inventory = max_inventory
        self.order_size = order_size


        self.inventory = 0
        self.cash = 0.0
        self.trades = 0
        self.pnl_history: List[float] = []

    def reset(self):
        self.inventory = 0
        self.cash = 0.0
        self.trades = 0
        self.pnl_history = []

    def quote(self, mid_price: float) -> Tuple[float, float]:
        bid = mid_price - self.delta_bid
        ask = mid_price + self.delta_ask
        return bid, ask

    def on_fill(self, side: str, price: float, volume: int,
                current_time: float = 0.0):
        if side == 'buy':
            self.inventory += volume
            self.cash -= price * volume
        else:
            self.inventory -= volume
            self.cash += price * volume
        self.trades += 1


        if abs(self.inventory) > self.max_inventory:

            penalty = self.inventory_penalty * (self.inventory ** 2)
            self.cash -= penalty

    def mark_to_market_pnl(self, current_price: float) -> float:
        return self.cash + self.inventory * current_price

    def inventory_risk_penalty(self) -> float:
        return self.inventory_penalty * (
            (self.inventory - self.inventory_target) ** 2
        )

    def get_params(self) -> np.ndarray:
        return np.array([self.delta_bid, self.delta_ask,
                         self.inventory_penalty], dtype=float)

    def set_params(self, params: np.ndarray):
        self.delta_bid = max(params[0], 1e-6)
        self.delta_ask = max(params[1], 1e-6)
        self.inventory_penalty = max(params[2], 0.0)


class BacktestEngine:

    def __init__(self,
                 price_path: np.ndarray,
                 time_grid: np.ndarray,
                 arrival_intensity: float = 50.0,
                 fill_probability_model: str = "exponential"):
        self.price_path = price_path
        self.time_grid = time_grid
        self.arrival_intensity = arrival_intensity
        self.fill_model = fill_probability_model

    def run(self, strategy: MarketMakingStrategy,
            seed: Optional[int] = None) -> dict:
        if seed is not None:
            np.random.seed(seed)

        strategy.reset()
        pnl_series = []
        dt = self.time_grid[1] - self.time_grid[0] if len(self.time_grid) > 1 else 1.0

        for i, mid in enumerate(self.price_path):
            t = self.time_grid[i]


            bid, ask = strategy.quote(mid)


            n_arrivals = np.random.poisson(self.arrival_intensity * dt)
            for _ in range(n_arrivals):

                if self.fill_model == "exponential":
                    p_fill_bid = np.exp(-strategy.delta_bid * 10.0)
                    p_fill_ask = np.exp(-strategy.delta_ask * 10.0)
                else:
                    p_fill_bid = max(0.0, 1.0 - strategy.delta_bid * 5.0)
                    p_fill_ask = max(0.0, 1.0 - strategy.delta_ask * 5.0)


                if np.random.uniform() < p_fill_bid:
                    if abs(strategy.inventory + strategy.order_size) <= strategy.max_inventory:
                        strategy.on_fill('buy', bid, strategy.order_size, t)


                if np.random.uniform() < p_fill_ask:
                    if abs(strategy.inventory - strategy.order_size) <= strategy.max_inventory:
                        strategy.on_fill('sell', ask, strategy.order_size, t)


            penalty = strategy.inventory_risk_penalty() * dt
            strategy.cash -= penalty


            mtm = strategy.mark_to_market_pnl(mid)
            pnl_series.append(mtm)

        pnl_arr = np.array(pnl_series)
        returns = np.diff(pnl_arr)

        metrics = self._compute_metrics(pnl_arr, returns)
        metrics['final_inventory'] = strategy.inventory
        metrics['total_trades'] = strategy.trades
        return metrics

    @staticmethod
    def _compute_metrics(pnl: np.ndarray, returns: np.ndarray) -> dict:
        total_return = pnl[-1] - pnl[0] if len(pnl) > 0 else 0.0

        if len(returns) > 0 and np.std(returns) > 1e-12:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 23400)


            sharpe_simple = np.mean(returns) / (np.std(returns) + 1e-12)
        else:
            sharpe = 0.0
            sharpe_simple = 0.0


        cumulative = pnl - pnl[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        return {
            'total_pnl': total_return,
            'sharpe_ratio': sharpe_simple,
            'max_drawdown': max_drawdown,
            'final_pnl': pnl[-1] if len(pnl) > 0 else 0.0,
        }


class StrategyOptimizer:

    def __init__(self,
                 backtest_engine: BacktestEngine,
                 n_epochs: int = 30,
                 learning_rate: float = 0.01,
                 batch_size: int = 5,
                 use_adagrad: bool = True):
        self.engine = backtest_engine
        self.n_epochs = n_epochs
        self.lr0 = learning_rate
        self.batch_size = batch_size
        self.use_adagrad = use_adagrad

    def objective(self, params: np.ndarray,
                  n_simulations: int = 5) -> float:
        strategy = MarketMakingStrategy()
        strategy.set_params(params)

        sharpe_vals = []
        for s in range(n_simulations):
            metrics = self.engine.run(strategy, seed=100 + s)
            sharpe_vals.append(-metrics['sharpe_ratio'])
            strategy.reset()

        return np.mean(sharpe_vals)

    def sgd_optimize(self, init_params: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        theta = init_params.copy()
        history = []
        G = np.zeros_like(theta)
        eps = 1e-5

        for epoch in range(self.n_epochs):
            grad = np.zeros_like(theta)
            for i in range(len(theta)):
                theta_plus = theta.copy()
                theta_minus = theta.copy()
                theta_plus[i] += eps
                theta_minus[i] -= eps
                grad[i] = (self.objective(theta_plus, self.batch_size)
                           - self.objective(theta_minus, self.batch_size)) / (2 * eps)


            if self.use_adagrad:
                G += grad ** 2
                alpha = self.lr0 / (np.sqrt(G) + 1e-8)
            else:
                alpha = np.full_like(theta, self.lr0)

            theta = theta - alpha * grad


            theta[0] = np.clip(theta[0], 1e-4, 1.0)
            theta[1] = np.clip(theta[1], 1e-4, 1.0)
            theta[2] = np.clip(theta[2], 0.0, 0.1)

            obj_val = self.objective(theta, 3)
            history.append(obj_val)

        return theta, history

    def brent_line_search(self,
                          f: Callable[[float], float],
                          a: float, b: float,
                          tol: float = 1e-6,
                          max_iter: int = 100) -> Tuple[float, float]:
        c_gr = 0.5 * (3.0 - np.sqrt(5.0))

        v = a + c_gr * (b - a)
        w = v
        x = v
        e = 0.0

        fx = f(x)
        fv = fx
        fw = fx

        for _ in range(max_iter):
            midpoint = 0.5 * (a + b)
            tol1 = np.sqrt(np.finfo(float).eps) * abs(x) + tol / 3.0
            tol2 = 2.0 * tol1

            if abs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
                return x, fx


            if abs(e) <= tol1:
                if x >= midpoint:
                    e = a - x
                else:
                    e = b - x
                d = c_gr * e
            else:

                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p = -p
                q = abs(q)
                r_temp = e
                e = d

                if (abs(0.5 * q * r_temp) <= abs(p) or
                        p <= q * (a - x) or p >= q * (b - x)):
                    if x >= midpoint:
                        e = a - x
                    else:
                        e = b - x
                    d = c_gr * e
                else:
                    d = p / q
                    u = x + d
                    if (u - a) < tol2 or (b - u) < tol2:
                        d = tol1 * np.sign(midpoint - x)

            if abs(d) >= tol1:
                u = x + d
            else:
                u = x + tol1 * np.sign(d)

            fu = f(u)

            if fu <= fx:
                if u >= x:
                    a = x
                else:
                    b = x
                v = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu

        return x, fx

    def optimize_spread(self, fixed_eta: float = 0.001) -> Tuple[float, float]:
        def f(delta):
            params = np.array([delta, delta, fixed_eta])
            return self.objective(params, n_simulations=3)

        best_delta, best_obj = self.brent_line_search(f, 1e-4, 1.0)
        return best_delta, best_obj
