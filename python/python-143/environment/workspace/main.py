
import numpy as np
import time

from price_dynamics import (
    OrnsteinUhlenbeck, StiffRelaxation,
    StabilityAnalysis, ParameterSweep
)
from market_simulator import (
    MarketSimulator, MarketEvent, OrderType, MarketState
)
from order_book_engine import (
    LimitOrderBook, LOBInterpolator, LOBGeometryAnalyzer
)
from strategy_optimizer import (
    MarketMakingStrategy, BacktestEngine, StrategyOptimizer
)
from risk_engine import (
    CovarianceEstimator, MinimumVariancePortfolio,
    RiskMetrics, RiskGeometry
)
from special_functions import SpecialFunctions
from data_compression import DictionaryEncoder
from chaos_analysis import ChaosAnalyzer
from numerical_integration import (
    FeketeQuadrature, FinancialExpectation, MultidimensionalQuadrature
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_price_dynamics():
    print_section("模块1: 多尺度价格动力学 (OU过程 + 刚性松弛)")













    raise NotImplementedError("Hole 3: 需补全 demo_price_dynamics 的完整实现")



def demo_market_simulation():
    print_section("模块2: 离散事件市场模拟 (泊松到达 + 状态机)")

    sim = MarketSimulator(duration_seconds=10.0, base_price=100.0, seed=42)
    events, sm = sim.run()

    print(f"[市场模拟] 生成事件数: {len(events)}")
    print(f"[市场模拟] 状态切换次数: {sm.cycle_count}")
    print(f"[市场模拟] 最终市场状态: {sm.state.name}")

    vwap = MarketSimulator.compute_vwap(events)
    imbalance = MarketSimulator.compute_order_imbalance(events)
    print(f"[市场统计] VWAP={vwap:.4f}, 订单不平衡度={imbalance:.4f}")


    type_counts = {}
    for e in events:
        type_counts[e.order_type.name] = type_counts.get(e.order_type.name, 0) + 1
    print(f"[事件分布] {type_counts}")

    return events


def demo_order_book(events):
    print_section("模块3: 限价订单簿 (LOB) 引擎与深度曲面插值")

    lob = LimitOrderBook(base_price=100.0, tick_size=0.01, max_levels=100)


    for e in events[:500]:
        if e.order_type == OrderType.LIMIT_BUY and e.price is not None:
            lob.add_order(e.price, e.volume, is_bid=True)
        elif e.order_type == OrderType.LIMIT_SELL and e.price is not None:
            lob.add_order(e.price, e.volume, is_bid=False)
        elif e.order_type == OrderType.CANCEL and e.price is not None:

            lob.cancel_order(e.price, e.volume, is_bid=(np.random.random() < 0.5))

    bb, ba, bbv, bav = lob.best_quotes()
    print(f"[LOB] 最优买价={bb}, 最优卖价={ba}, 价差={ba-bb if bb and ba else None}")


    p_bid, d_bid = lob.depth_profile("bid", n_levels=10)
    p_ask, d_ask = lob.depth_profile("ask", n_levels=10)
    print(f"[LOB] 前5档买单深度: {d_bid[:5] if len(d_bid)>=5 else d_bid}")
    print(f"[LOB] 前5档卖单深度: {d_ask[:5] if len(d_ask)>=5 else d_ask}")


    if len(p_bid) >= 3:
        interp = LOBInterpolator()
        p_query = np.linspace(p_bid.min(), p_bid.max(), 20)
        d_interp = interp.quadratic_interpolate(p_bid, d_bid.astype(float), p_query)
        print(f"[LOB插值] 二次插值后深度范围: [{d_interp.min():.2f}, {d_interp.max():.2f}]")


    geo = LOBGeometryAnalyzer()
    if len(p_bid) >= 3:
        convex_ratio = geo.lob_convexity(p_bid, d_bid.astype(float))
        conc = geo.depth_concentration(d_bid)
        print(f"[LOB几何] 深度凸性比率={convex_ratio:.4f}, 集中度={conc:.4f}")


    if bb is not None and ba is not None:
        exec_vol, avg_price = lob.execute_market_order(10, is_buy=True)
        print(f"[LOB成交] 买入10单位, 实际成交{exec_vol}, 均价={avg_price:.4f}")

    return lob


def demo_strategy_optimization(price_path):
    print_section("模块4: 高频做市策略优化 (SGD + Brent线搜索)")

    time_grid = np.linspace(0.0, 1.0, len(price_path))
    engine = BacktestEngine(
        price_path=price_path,
        time_grid=time_grid,
        arrival_intensity=30.0,
        fill_probability_model="exponential"
    )


    strategy = MarketMakingStrategy(delta_bid=0.05, delta_ask=0.05)
    metrics = engine.run(strategy, seed=123)
    print(f"[初始策略] PnL={metrics['total_pnl']:.4f}, "
          f"Sharpe={metrics['sharpe_ratio']:.4f}, MDD={metrics['max_drawdown']:.4f}")


    optimizer = StrategyOptimizer(
        backtest_engine=engine,
        n_epochs=10,
        learning_rate=0.005,
        batch_size=3,
        use_adagrad=True
    )
    init_params = np.array([0.05, 0.05, 0.001])
    best_params, history = optimizer.sgd_optimize(init_params)
    print(f"[SGD优化] 最优参数: δ_bid={best_params[0]:.4f}, "
          f"δ_ask={best_params[1]:.4f}, η={best_params[2]:.6f}")
    print(f"[SGD优化] 目标函数历史: {[f'{h:.4f}' for h in history[:5]]} ...")


    best_delta, best_obj = optimizer.optimize_spread(fixed_eta=0.001)
    print(f"[Brent线搜索] 最优对称价差 δ={best_delta:.6f}, 目标值={best_obj:.4f}")


    strategy_opt = MarketMakingStrategy()
    strategy_opt.set_params(best_params)
    metrics_opt = engine.run(strategy_opt, seed=456)
    print(f"[优化后策略] PnL={metrics_opt['total_pnl']:.4f}, "
          f"Sharpe={metrics_opt['sharpe_ratio']:.4f}, MDD={metrics_opt['max_drawdown']:.4f}")

    return metrics_opt


def demo_risk_analysis(price_path):
    print_section("模块5: 风险引擎 (协方差估计 + 凸几何)")


    returns = np.diff(price_path)
    n_assets = 5
    returns_multi = np.column_stack([
        returns + np.random.normal(0, 0.1, len(returns)) for _ in range(n_assets)
    ])


    cov_est = CovarianceEstimator(n_assets=n_assets, decay=0.94)
    for t in range(len(returns_multi)):
        cov_est.update(returns_multi[t])

    corr = cov_est.get_correlation()
    print(f"[协方差估计] 最终相关系数矩阵:\n{np.round(corr, 3)}")


    mvp = MinimumVariancePortfolio(cov_est)
    w_mv = mvp.solve()
    var_mv = mvp.portfolio_variance(w_mv)
    print(f"[最小方差组合] 权重={np.round(w_mv, 3)}, 组合方差={var_mv:.6f}")


    rm = RiskMetrics()
    var_95 = rm.value_at_risk(returns)
    es_95 = rm.expected_shortfall(returns)
    cum = np.cumsum(returns)
    mdd = rm.max_drawdown(cum)
    calmar = rm.calmar_ratio(returns, cum)
    print(f"[风险指标] VaR(95%)={var_95:.4f}, ES(95%)={es_95:.4f}, MDD={mdd:.4f}, Calmar={calmar:.4f}")


    rg = RiskGeometry()

    scatter = np.column_stack([
        np.mean(returns_multi, axis=0),
        np.std(returns_multi, axis=0)
    ])
    area = rg.convex_hull_area_2d(scatter)
    print(f"[凸几何] 收益-风险平面凸包面积={area:.6f}")

    return var_95, es_95


def demo_special_functions():
    print_section("模块6: 金融特殊函数 (Ci/Si/正态CDF)")

    sf = SpecialFunctions()


    test_vals = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
    for xv in test_vals:
        civ = sf.ci(xv)
        print(f"[Ci({xv})] = {civ:.8f}")


    z_vals = [-3.0, -1.0, 0.0, 1.0, 3.0]
    for z in z_vals:
        ncdf = sf.normal_cdf(z)
        print(f"[N({z})] = {ncdf:.8f}")


    delta = sf.black_scholes_delta(S=100.0, K=100.0, T=0.25,
                                   r=0.05, sigma=0.2, option_type="call")
    print(f"[BS Delta] ATM Call Delta = {delta:.6f}")

    return sf


def demo_data_compression(events):
    print_section("模块7: 高频数据字典编码压缩")


    n = min(len(events), 1000)
    price_changes = np.zeros(n)
    volumes = np.zeros(n, dtype=int)
    sides = np.zeros(n, dtype=int)
    types = np.zeros(n, dtype=int)

    base_price = 100.0
    for i in range(n):
        e = events[i]
        price_changes[i] = (e.price - base_price) if e.price is not None else 0.0
        volumes[i] = e.volume
        sides[i] = 0 if e.order_type in (OrderType.MARKET_BUY, OrderType.LIMIT_BUY) else 1
        types[i] = e.order_type.value

    encoder = DictionaryEncoder(price_tick_size=0.01, volume_bucket_size=2)
    encoded = encoder.build_dictionary(price_changes, volumes, sides, types)
    rle = encoder.run_length_encode(encoded)

    print(f"[字典编码] 原始记录数={n}, 字典大小={len(encoder.dictionary)}, 编码后长度={len(encoded)}")
    print(f"[游程编码] RLE后长度={len(rle)}")
    entropy = encoder.compute_entropy()
    print(f"[信息熵] 编码熵率={entropy:.4f} bits/symbol")
    cr = encoder.compression_ratio(original_size_bytes=n * 16)
    print(f"[压缩比] 估计压缩比={cr:.4f}x")

    return encoder


def demo_chaos_analysis(price_path):
    print_section("模块8: 市场混沌与分形分析")

    analyzer = ChaosAnalyzer()
    metrics = analyzer.analyze_price_path(price_path)

    print(f"[分形分析] 盒维数={metrics['box_dimension']:.4f}")
    print(f"[混沌分析] 最大Lyapunov指数={metrics['lyapunov_max']:.6f}")
    print(f"[记忆性分析] Hurst指数={metrics['hurst']:.4f}")
    print(f"[自相关分析] 收益率一阶自相关={metrics['returns_autocorr']:.4f}")

    regime = analyzer.regime_classification(metrics)
    print(f"[Regime分类] 当前市场处于: {regime}")

    return metrics


def demo_numerical_integration():
    print_section("模块9: Fekete点数值积分与金融期望")


    fq = FeketeQuadrature(-1.0, 1.0)
    result = fq.integrate(lambda x: np.exp(x), m=10)
    exact = np.exp(1.0) - np.exp(-1.0)
    print(f"[一维积分] ∫_{'{-1}'}^{'{1}'} e^x dx ≈ {result:.8f}, 精确值={exact:.8f}, 误差={abs(result-exact):.2e}")


    result_2d = MultidimensionalQuadrature.tensor_product_2d(
        lambda x, y: x ** 2 * y ** 2,
        m1=8, m2=8, a1=-1.0, b1=1.0, a2=-1.0, b2=1.0
    )
    exact_2d = 4.0 / 9.0
    print(f"[二维积分] ∫∫ x²y² dxdy ≈ {result_2d:.8f}, 精确值={exact_2d:.8f}, 误差={abs(result_2d-exact_2d):.2e}")


    payoff = lambda x: np.maximum(0.0, x - 1.0)
    exp_payoff = FinancialExpectation.expected_payoff_fekete(payoff, m=20, a=-5.0, b=5.0)

    from scipy.stats import norm
    exact_payoff = norm.pdf(1.0) - 1.0 * norm.cdf(-1.0)
    print(f"[金融期望] E[max(0,X-1)] ≈ {exp_payoff:.8f}, 精确值={exact_payoff:.8f}, 误差={abs(exp_payoff-exact_payoff):.2e}")

    return result


def main():
    print("=" * 70)
    print("  高频交易策略回测系统 — 博士级科研代码合成项目")
    print("  领域: 金融工程 — 高频交易策略回测与参数优化")
    print("=" * 70)
    print(f"\n开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()


    price_path = demo_price_dynamics()


    events = demo_market_simulation()


    lob = demo_order_book(events)


    metrics_opt = demo_strategy_optimization(price_path)


    var_95, es_95 = demo_risk_analysis(price_path)


    sf = demo_special_functions()


    encoder = demo_data_compression(events)


    chaos_metrics = demo_chaos_analysis(price_path)


    int_result = demo_numerical_integration()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  所有模块执行完毕!")
    print(f"  总耗时: {elapsed:.3f} 秒")
    print("=" * 70)


    print("\n[系统汇总]")
    print(f"  - 价格路径长度: {len(price_path)}")
    print(f"  - 模拟市场事件数: {len(events)}")
    print(f"  - 策略优化后 Sharpe: {metrics_opt.get('sharpe_ratio', 0):.4f}")
    print(f"  - VaR(95%): {var_95:.4f}")
    print(f"  - ES(95%): {es_95:.4f}")
    print(f"  - 市场Regime: {ChaosAnalyzer().regime_classification(chaos_metrics)}")
    print(f"  - 数据字典大小: {len(encoder.dictionary)}")
    print(f"  - Fekete积分精度: {abs(int_result - (np.exp(1.0) - np.exp(-1.0))):.2e}")

    print("\n系统正常退出.")


if __name__ == "__main__":
    main()
