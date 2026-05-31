"""
strategy_optimizer.py
=====================
高频做市策略的参数优化与回测

本模块基于以下种子项目融合:
- 478_gradient_descent: 梯度下降优化 → 策略超参数的随机梯度下降优化
- 695_local_min_rc: Brent一维局部最小值搜索 → 最优买卖价差的线搜索

核心数学模型:
--------------
1.  高频做市策略 (Market Making):
    做市商同时在最优买价 P_bid 挂买单, 在最优卖价 P_ask 挂卖单.
    设挂单价差偏移为 δ_bid, δ_ask (相对于中间价 P_mid 的偏移):
        P_post_bid = P_mid - δ_bid
        P_post_ask = P_mid + δ_ask
    当价格触及挂单时成交, 做市商赚取买卖价差.

2.  库存风险约束:
    设 I_t 为 t 时刻的库存 (净持仓).
    库存偏离目标 I* 的惩罚项:
        R_inv(I_t) = η (I_t - I*)²
    其中 η > 0 为库存厌恶系数.
    对应的 Hamilton-Jacobi-Bellman (HJB) 方程:
        0 = max_{δ} { λ(δ) δ - η I² + ∂V/∂t + μ ∂V/∂S
                     + 0.5 σ² ∂²V/∂S² + λ(δ) [V(S, I±1) - V(S, I)] }
    其中 λ(δ) = A exp(-k δ) 为到达强度 (Poisson 强度).

3.  策略收益泛函:
    对回测路径 {S_t, I_t}_{t=0}^T:
        J(θ) = E[ Σ_t (成交收益_t - 库存惩罚_t -  adverse selection 损失_t) ]
    其中 θ = (δ_bid, δ_ask, η, k) 为策略参数.

4.  随机梯度下降 (SGD) 优化:
    对目标函数 J(θ), 采用小批量梯度估计:
        g_t = ∇_θ J_t(θ_t)
        θ_{t+1} = θ_t - α g_t
    其中 α 为学习率, 采用 AdaGrad 自适应调整:
        G_t = G_{t-1} + g_t ⊙ g_t
        α_t = α_0 / (√G_t + ε)

5.  一维线搜索 (基于 695_local_min_rc 的 Brent 方法思想):
    对固定其他参数时的单参数优化, 采用黄金分割 + 抛物线插值:
        φ(ρ) = (3 - √5) / 2 ≈ 0.381966  (黄金分割比倒数)
        迭代直到区间长度 |b-a| < tol.
    抛物线插值步骤:
        p = (x-v)²(f_x-f_w) - (x-w)²(f_x-f_v)
        q = 2[(x-v)(f_x-f_w) - (x-w)(f_x-f_v)]
        u = x - p/q
"""

import numpy as np
from typing import Callable, Tuple, List, Optional


class MarketMakingStrategy:
    """
    高频做市策略核心.
    """

    def __init__(self,
                 delta_bid: float = 0.02,
                 delta_ask: float = 0.02,
                 inventory_target: int = 0,
                 inventory_penalty: float = 0.001,
                 max_inventory: int = 100,
                 order_size: int = 1):
        """
        Parameters
        ----------
        delta_bid, delta_ask : float
            相对于中间价的挂单偏移.
        inventory_target : int
            目标库存水平.
        inventory_penalty : float
            库存偏离惩罚系数 η.
        max_inventory : int
            最大允许库存 (边界约束).
        order_size : int
            单笔挂单量.
        """
        self.delta_bid = delta_bid
        self.delta_ask = delta_ask
        self.inventory_target = inventory_target
        self.inventory_penalty = inventory_penalty
        self.max_inventory = max_inventory
        self.order_size = order_size

        # 状态
        self.inventory = 0
        self.cash = 0.0
        self.trades = 0
        self.pnl_history: List[float] = []

    def reset(self):
        """重置策略状态."""
        self.inventory = 0
        self.cash = 0.0
        self.trades = 0
        self.pnl_history = []

    def quote(self, mid_price: float) -> Tuple[float, float]:
        """
        生成做市报价.

        Returns
        -------
        bid_price, ask_price : float
        """
        bid = mid_price - self.delta_bid
        ask = mid_price + self.delta_ask
        return bid, ask

    def on_fill(self, side: str, price: float, volume: int,
                current_time: float = 0.0):
        """
        订单成交回调.
        side: 'buy' 表示我方买入 (吃掉 ask), 'sell' 表示我方卖出.
        """
        if side == 'buy':
            self.inventory += volume
            self.cash -= price * volume
        else:
            self.inventory -= volume
            self.cash += price * volume
        self.trades += 1

        # 边界检查
        if abs(self.inventory) > self.max_inventory:
            # 强制平仓惩罚
            penalty = self.inventory_penalty * (self.inventory ** 2)
            self.cash -= penalty

    def mark_to_market_pnl(self, current_price: float) -> float:
        """
        按市价计算的浮动盈亏:
            PnL = Cash + Inventory * Current_Price
        """
        return self.cash + self.inventory * current_price

    def inventory_risk_penalty(self) -> float:
        """当前库存惩罚."""
        return self.inventory_penalty * (
            (self.inventory - self.inventory_target) ** 2
        )

    def get_params(self) -> np.ndarray:
        """返回参数向量."""
        return np.array([self.delta_bid, self.delta_ask,
                         self.inventory_penalty], dtype=float)

    def set_params(self, params: np.ndarray):
        """从向量设置参数."""
        self.delta_bid = max(params[0], 1e-6)
        self.delta_ask = max(params[1], 1e-6)
        self.inventory_penalty = max(params[2], 0.0)


class BacktestEngine:
    """
    回测引擎, 在给定价格路径上评估策略性能.
    """

    def __init__(self,
                 price_path: np.ndarray,
                 time_grid: np.ndarray,
                 arrival_intensity: float = 50.0,
                 fill_probability_model: str = "exponential"):
        """
        Parameters
        ----------
        price_path : np.ndarray
            价格路径 S_t.
        time_grid : np.ndarray
            对应时间网格.
        arrival_intensity : float
            对手方订单到达强度 (决定成交概率).
        fill_probability_model : str
            "exponential" 或 "linear".
        """
        self.price_path = price_path
        self.time_grid = time_grid
        self.arrival_intensity = arrival_intensity
        self.fill_model = fill_probability_model

    def run(self, strategy: MarketMakingStrategy,
            seed: Optional[int] = None) -> dict:
        """
        执行回测.

        Returns
        -------
        metrics : dict
            包含夏普比率、最大回撤、总收益等指标.
        """
        if seed is not None:
            np.random.seed(seed)

        strategy.reset()
        pnl_series = []
        dt = self.time_grid[1] - self.time_grid[0] if len(self.time_grid) > 1 else 1.0

        for i, mid in enumerate(self.price_path):
            t = self.time_grid[i]

            # 策略报价
            bid, ask = strategy.quote(mid)

            # 模拟对手方到达与成交
            n_arrivals = np.random.poisson(self.arrival_intensity * dt)
            for _ in range(n_arrivals):
                # 成交概率取决于报价偏移
                if self.fill_model == "exponential":
                    p_fill_bid = np.exp(-strategy.delta_bid * 10.0)
                    p_fill_ask = np.exp(-strategy.delta_ask * 10.0)
                else:
                    p_fill_bid = max(0.0, 1.0 - strategy.delta_bid * 5.0)
                    p_fill_ask = max(0.0, 1.0 - strategy.delta_ask * 5.0)

                # 买入成交 (对手方 market sell, 我方 buy)
                if np.random.uniform() < p_fill_bid:
                    if abs(strategy.inventory + strategy.order_size) <= strategy.max_inventory:
                        strategy.on_fill('buy', bid, strategy.order_size, t)

                # 卖出成交 (对手方 market buy, 我方 sell)
                if np.random.uniform() < p_fill_ask:
                    if abs(strategy.inventory - strategy.order_size) <= strategy.max_inventory:
                        strategy.on_fill('sell', ask, strategy.order_size, t)

            # 库存惩罚 (实时扣除)
            penalty = strategy.inventory_risk_penalty() * dt
            strategy.cash -= penalty

            # 记录 PnL
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
        """计算回测绩效指标."""
        total_return = pnl[-1] - pnl[0] if len(pnl) > 0 else 0.0

        if len(returns) > 0 and np.std(returns) > 1e-12:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 23400)
            # 假设每秒一个点, 日频 Sharpe 年化因子 √(~6M)
            # 简化: 使用路径内 Sharpe
            sharpe_simple = np.mean(returns) / (np.std(returns) + 1e-12)
        else:
            sharpe = 0.0
            sharpe_simple = 0.0

        # 最大回撤
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
    """
    策略参数优化器, 基于梯度下降与 Brent 线搜索.
    """

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
        """
        目标函数: 负平均 Sharpe 比率 (最小化).
        """
        strategy = MarketMakingStrategy()
        strategy.set_params(params)

        sharpe_vals = []
        for s in range(n_simulations):
            metrics = self.engine.run(strategy, seed=100 + s)
            sharpe_vals.append(-metrics['sharpe_ratio'])  # 负号用于最小化
            strategy.reset()

        return np.mean(sharpe_vals)

    def sgd_optimize(self, init_params: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        随机梯度下降优化.

        对参数 θ = (δ_bid, δ_ask, η):
            g_t ≈ [J(θ+εe_i) - J(θ-εe_i)] / (2ε)
            θ_{t+1} = θ_t - α_t g_t
        """
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

            # AdaGrad
            if self.use_adagrad:
                G += grad ** 2
                alpha = self.lr0 / (np.sqrt(G) + 1e-8)
            else:
                alpha = np.full_like(theta, self.lr0)

            theta = theta - alpha * grad

            # 投影到可行域
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
        """
        Brent 方法一维最小化 (基于 695_local_min_rc 思想).
        结合黄金分割搜索与抛物线插值.

        算法:
        1.  初始化: c = (3-√5)/2
        2.  取试探点 v = a + c(b-a), w = v, x = v
        3.  循环:
            a) 若 |x-midpoint| ≤ tol 停止
            b) 尝试抛物线插值求极小值点 u
            c) 若插值不可靠, 改用黄金分割步
            d) 更新区间 [a,b] 和点集 {v,w,x}
        """
        c_gr = 0.5 * (3.0 - np.sqrt(5.0))  # ≈0.381966

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

            # 判断是否使用黄金分割
            if abs(e) <= tol1:
                if x >= midpoint:
                    e = a - x
                else:
                    e = b - x
                d = c_gr * e
            else:
                # 抛物线插值
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
        """
        固定库存惩罚, 仅优化买卖价差 (一维问题).
        假设对称报价 δ_bid = δ_ask = δ.
        """
        def f(delta):
            params = np.array([delta, delta, fixed_eta])
            return self.objective(params, n_simulations=3)

        best_delta, best_obj = self.brent_line_search(f, 1e-4, 1.0)
        return best_delta, best_obj
