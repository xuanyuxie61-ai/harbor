"""
main.py
=======
高频交易策略回测系统 — 统一入口

本系统基于15个科研代码项目的核心算法, 融合构建了一个面向金融工程前沿的
博士级高频交易策略回测与优化平台.

运行方式:
    python main.py

零参数运行, 自动执行完整的回测流程.
"""

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
    """模块1: 多尺度价格动力学模拟与稳定性分析."""
    print_section("模块1: 多尺度价格动力学 (OU过程 + 刚性松弛)")

    # 1a. OU 过程模拟
    ou = OrnsteinUhlenbeck(
        kappa=5.0, mu=100.0, sigma=2.0,
        s0=100.0, t_max=1.0, n_steps=5000, seed=42
    )
    t, path_rk2 = ou.simulate_rk2()
    _, path_exact = ou.simulate_exact_milstein()

    # 强误差估计
    strong_error = np.max(np.abs(path_rk2 - path_exact))
    print(f"[OU模拟] RK2 vs 精确解的强误差: {strong_error:.6f}")

    exp_final, var_final = ou.exact_solution(np.array([ou.t_max]))
    print(f"[OU理论] t=T 时期望={exp_final[0]:.4f}, 方差={var_final[0]:.6f}")
    print(f"[OU样本] t=T 时价格={path_exact[-1]:.4f}")

    # 1b. 刚性松弛
    stiff = StiffRelaxation(lam=100.0, omega=20.0, y0=0.0,
                            t_max=1.0, n_steps=2000)
    t_s, y_rk2 = stiff.solve_rk2()
    _, y_exact = stiff.solve_exact()
    stiff_error = np.max(np.abs(y_rk2 - y_exact))
    print(f"[刚性ODE] RK2 vs 精确解误差: {stiff_error:.6e}")

    # 1c. 稳定性分析
    stab = StabilityAnalysis()
    h_max = stab.maximum_stable_step(lambda_max=100.0)
    print(f"[稳定性] 对 λ=100 的刚性ODE, RK2最大稳定步长 h_max={h_max:.6f}")
    is_st = stab.is_stable(-1.5, 0.5)
    print(f"[稳定性] 点 z=-1.5+0.5i 在稳定区内: {is_st}")

    # 1d. 参数扫描
    kappa_vals = np.array([1.0, 5.0, 10.0, 20.0])
    sigma_vals = np.array([0.5, 1.0, 2.0, 4.0])
    sweep = ParameterSweep(kappa_vals, sigma_vals, mu=100.0,
                           t_max=0.5, n_steps=2000)
    half_lives = sweep.sweep_half_life()
    var_infs = sweep.sweep_stationary_variance()
    print(f"[参数扫描] 半衰期矩阵 (ln2/κ):\n{half_lives}")
    print(f"[参数扫描] 稳态方差矩阵 (σ²/2κ):\n{var_infs}")

    return path_exact


def demo_market_simulation():
    """模块2: 离散事件市场模拟."""
    print_section("模块2: 离散事件市场模拟 (泊松到达 + 状态机)")

    sim = MarketSimulator(duration_seconds=10.0, base_price=100.0, seed=42)
    events, sm = sim.run()

    print(f"[市场模拟] 生成事件数: {len(events)}")
    print(f"[市场模拟] 状态切换次数: {sm.cycle_count}")
    print(f"[市场模拟] 最终市场状态: {sm.state.name}")

    vwap = MarketSimulator.compute_vwap(events)
    imbalance = MarketSimulator.compute_order_imbalance(events)
    print(f"[市场统计] VWAP={vwap:.4f}, 订单不平衡度={imbalance:.4f}")

    # 统计各类型事件数
    type_counts = {}
    for e in events:
        type_counts[e.order_type.name] = type_counts.get(e.order_type.name, 0) + 1
    print(f"[事件分布] {type_counts}")

    return events


def demo_order_book(events):
    """模块3: 限价订单簿引擎与几何分析."""
    print_section("模块3: 限价订单簿 (LOB) 引擎与深度曲面插值")

    lob = LimitOrderBook(base_price=100.0, tick_size=0.01, max_levels=100)

    # 用模拟事件填充订单簿
    for e in events[:500]:
        if e.order_type == OrderType.LIMIT_BUY and e.price is not None:
            lob.add_order(e.price, e.volume, is_bid=True)
        elif e.order_type == OrderType.LIMIT_SELL and e.price is not None:
            lob.add_order(e.price, e.volume, is_bid=False)
        elif e.order_type == OrderType.CANCEL and e.price is not None:
            # 简化: 随机撤买单或卖单
            lob.cancel_order(e.price, e.volume, is_bid=(np.random.random() < 0.5))

    bb, ba, bbv, bav = lob.best_quotes()
    print(f"[LOB] 最优买价={bb}, 最优卖价={ba}, 价差={ba-bb if bb and ba else None}")

    # 深度剖面
    p_bid, d_bid = lob.depth_profile("bid", n_levels=10)
    p_ask, d_ask = lob.depth_profile("ask", n_levels=10)
    print(f"[LOB] 前5档买单深度: {d_bid[:5] if len(d_bid)>=5 else d_bid}")
    print(f"[LOB] 前5档卖单深度: {d_ask[:5] if len(d_ask)>=5 else d_ask}")

    # 二次插值
    if len(p_bid) >= 3:
        interp = LOBInterpolator()
        p_query = np.linspace(p_bid.min(), p_bid.max(), 20)
        d_interp = interp.quadratic_interpolate(p_bid, d_bid.astype(float), p_query)
        print(f"[LOB插值] 二次插值后深度范围: [{d_interp.min():.2f}, {d_interp.max():.2f}]")

    # 几何分析
    geo = LOBGeometryAnalyzer()
    if len(p_bid) >= 3:
        convex_ratio = geo.lob_convexity(p_bid, d_bid.astype(float))
        conc = geo.depth_concentration(d_bid)
        print(f"[LOB几何] 深度凸性比率={convex_ratio:.4f}, 集中度={conc:.4f}")

    # 执行市价单
    if bb is not None and ba is not None:
        exec_vol, avg_price = lob.execute_market_order(10, is_buy=True)
        print(f"[LOB成交] 买入10单位, 实际成交{exec_vol}, 均价={avg_price:.4f}")

    return lob


def demo_strategy_optimization(price_path):
    """模块4: 策略优化."""
    print_section("模块4: 高频做市策略优化 (SGD + Brent线搜索)")

    time_grid = np.linspace(0.0, 1.0, len(price_path))
    engine = BacktestEngine(
        price_path=price_path,
        time_grid=time_grid,
        arrival_intensity=30.0,
        fill_probability_model="exponential"
    )

    # 初始策略
    strategy = MarketMakingStrategy(delta_bid=0.05, delta_ask=0.05)
    metrics = engine.run(strategy, seed=123)
    print(f"[初始策略] PnL={metrics['total_pnl']:.4f}, "
          f"Sharpe={metrics['sharpe_ratio']:.4f}, MDD={metrics['max_drawdown']:.4f}")

    # SGD 优化
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

    # Brent 线搜索 (对称价差)
    best_delta, best_obj = optimizer.optimize_spread(fixed_eta=0.001)
    print(f"[Brent线搜索] 最优对称价差 δ={best_delta:.6f}, 目标值={best_obj:.4f}")

    # 评估优化后策略
    strategy_opt = MarketMakingStrategy()
    strategy_opt.set_params(best_params)
    metrics_opt = engine.run(strategy_opt, seed=456)
    print(f"[优化后策略] PnL={metrics_opt['total_pnl']:.4f}, "
          f"Sharpe={metrics_opt['sharpe_ratio']:.4f}, MDD={metrics_opt['max_drawdown']:.4f}")

    return metrics_opt


def demo_risk_analysis(price_path):
    """模块5: 风险矩阵与凸几何分析."""
    print_section("模块5: 风险引擎 (协方差估计 + 凸几何)")

    # 构造多资产收益
    returns = np.diff(price_path)
    n_assets = 5
    returns_multi = np.column_stack([
        returns + np.random.normal(0, 0.1, len(returns)) for _ in range(n_assets)
    ])

    # EWMA 协方差估计
    cov_est = CovarianceEstimator(n_assets=n_assets, decay=0.94)
    for t in range(len(returns_multi)):
        cov_est.update(returns_multi[t])

    corr = cov_est.get_correlation()
    print(f"[协方差估计] 最终相关系数矩阵:\n{np.round(corr, 3)}")

    # 最小方差组合
    mvp = MinimumVariancePortfolio(cov_est)
    w_mv = mvp.solve()
    var_mv = mvp.portfolio_variance(w_mv)
    print(f"[最小方差组合] 权重={np.round(w_mv, 3)}, 组合方差={var_mv:.6f}")

    # 风险指标
    rm = RiskMetrics()
    var_95 = rm.value_at_risk(returns)
    es_95 = rm.expected_shortfall(returns)
    cum = np.cumsum(returns)
    mdd = rm.max_drawdown(cum)
    calmar = rm.calmar_ratio(returns, cum)
    print(f"[风险指标] VaR(95%)={var_95:.4f}, ES(95%)={es_95:.4f}, MDD={mdd:.4f}, Calmar={calmar:.4f}")

    # 凸几何
    rg = RiskGeometry()
    # 构造 (收益, 风险) 散点
    scatter = np.column_stack([
        np.mean(returns_multi, axis=0),
        np.std(returns_multi, axis=0)
    ])
    area = rg.convex_hull_area_2d(scatter)
    print(f"[凸几何] 收益-风险平面凸包面积={area:.6f}")

    return var_95, es_95


def demo_special_functions():
    """模块6: 特殊函数."""
    print_section("模块6: 金融特殊函数 (Ci/Si/正态CDF)")

    sf = SpecialFunctions()

    # Ci 函数
    test_vals = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
    for xv in test_vals:
        civ = sf.ci(xv)
        print(f"[Ci({xv})] = {civ:.8f}")

    # 正态CDF
    z_vals = [-3.0, -1.0, 0.0, 1.0, 3.0]
    for z in z_vals:
        ncdf = sf.normal_cdf(z)
        print(f"[N({z})] = {ncdf:.8f}")

    # Black-Scholes Delta
    delta = sf.black_scholes_delta(S=100.0, K=100.0, T=0.25,
                                   r=0.05, sigma=0.2, option_type="call")
    print(f"[BS Delta] ATM Call Delta = {delta:.6f}")

    return sf


def demo_data_compression(events):
    """模块7: 高频数据字典编码压缩."""
    print_section("模块7: 高频数据字典编码压缩")

    # 提取事件特征
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
    """模块8: 混沌与分形分析."""
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
    """模块9: Fekete点数值积分."""
    print_section("模块9: Fekete点数值积分与金融期望")

    # 测试积分: ∫_{-1}^1 exp(x) dx = e - 1/e ≈ 2.350402
    fq = FeketeQuadrature(-1.0, 1.0)
    result = fq.integrate(lambda x: np.exp(x), m=10)
    exact = np.exp(1.0) - np.exp(-1.0)
    print(f"[一维积分] ∫_{'{-1}'}^{'{1}'} e^x dx ≈ {result:.8f}, 精确值={exact:.8f}, 误差={abs(result-exact):.2e}")

    # 二维张量积: ∫_{-1}^1 ∫_{-1}^1 x² y² dx dy = 4/9
    result_2d = MultidimensionalQuadrature.tensor_product_2d(
        lambda x, y: x ** 2 * y ** 2,
        m1=8, m2=8, a1=-1.0, b1=1.0, a2=-1.0, b2=1.0
    )
    exact_2d = 4.0 / 9.0
    print(f"[二维积分] ∫∫ x²y² dxdy ≈ {result_2d:.8f}, 精确值={exact_2d:.8f}, 误差={abs(result_2d-exact_2d):.2e}")

    # 金融期望: 计算标准正态下 E[max(0, X-1)]
    payoff = lambda x: np.maximum(0.0, x - 1.0)
    exp_payoff = FinancialExpectation.expected_payoff_fekete(payoff, m=20, a=-5.0, b=5.0)
    # 精确值: 对一个标准正态, E[max(0,X-1)] = φ(1) - 1*N(-1) ≈ 0.0833
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

    # 模块1: 价格动力学
    price_path = demo_price_dynamics()

    # 模块2: 市场模拟
    events = demo_market_simulation()

    # 模块3: 订单簿
    lob = demo_order_book(events)

    # 模块4: 策略优化
    metrics_opt = demo_strategy_optimization(price_path)

    # 模块5: 风险分析
    var_95, es_95 = demo_risk_analysis(price_path)

    # 模块6: 特殊函数
    sf = demo_special_functions()

    # 模块7: 数据压缩
    encoder = demo_data_compression(events)

    # 模块8: 混沌分析
    chaos_metrics = demo_chaos_analysis(price_path)

    # 模块9: 数值积分
    int_result = demo_numerical_integration()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  所有模块执行完毕!")
    print(f"  总耗时: {elapsed:.3f} 秒")
    print("=" * 70)

    # 最终汇总
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

# ================================================================
# 测试用例（50个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: OU精确解 - t=0时期望等于初始值 ----
ou = OrnsteinUhlenbeck(kappa=5.0, mu=100.0, sigma=2.0, s0=100.0, t_max=1.0, n_steps=1000, seed=42)
exp0, var0 = ou.exact_solution(np.array([0.0]))
assert abs(exp0[0] - 100.0) < 1e-10, '[TC01] OU t=0时期望应等于s0 FAILED'

# ---- TC02: OU精确解 - t足够大时期望趋近μ ----
exp_inf, var_inf = ou.exact_solution(np.array([100.0]))
assert abs(exp_inf[0] - 100.0) < 1e-10, '[TC02] OU大t时期望应趋近μ FAILED'

# ---- TC03: OU精确解 - 方差始终非负 ----
exp_mid, var_mid = ou.exact_solution(np.array([0.5]))
assert var_mid[0] >= 0.0, '[TC03] OU方差必须非负 FAILED'
assert var_mid[0] < 1.0, '[TC03] OU方差应在合理范围内 FAILED'

# ---- TC04: OU模拟可复现性 ----
ou1 = OrnsteinUhlenbeck(kappa=3.0, mu=50.0, sigma=1.0, s0=50.0, t_max=1.0, n_steps=500, seed=42)
t1, s1 = ou1.simulate_exact_milstein()
ou2 = OrnsteinUhlenbeck(kappa=3.0, mu=50.0, sigma=1.0, s0=50.0, t_max=1.0, n_steps=500, seed=42)
t2, s2 = ou2.simulate_exact_milstein()
assert np.allclose(s1, s2), '[TC04] 相同种子应产生相同模拟路径 FAILED'

# ---- TC05: StiffRelaxation精确解 - 初始条件验证 ----
stiff = StiffRelaxation(lam=50.0, omega=10.0, y0=1.0, t_max=0.5, n_steps=500)
t_s, y_exact = stiff.solve_exact()
assert len(y_exact) == 501, '[TC05] 精确解长度应为n_steps+1 FAILED'
assert abs(y_exact[0] - 1.0) < 1e-12, '[TC05] 初始条件 y(0)=y0 FAILED'

# ---- TC06: 稳定性分析 - 最大稳定步长计算 ----
h_max = StabilityAnalysis.maximum_stable_step(100.0)
assert abs(h_max - 0.02) < 1e-10, '[TC06] λ=100时最大稳定步长应为2/λ=0.02 FAILED'

# ---- TC07: 稳定性分析 - 稳定区域边界验证 ----
assert StabilityAnalysis.is_stable(-1.5, 0.0), '[TC07] z=-1.5应在绝对稳定区内 FAILED'
assert StabilityAnalysis.is_stable(-0.5, 0.5), '[TC07] z=-0.5+0.5i应在绝对稳定区内 FAILED'

# ---- TC08: 参数扫描 - 半衰期矩阵形状与值 ----
kappa_vals = np.array([1.0, 5.0, 10.0])
sigma_vals = np.array([0.5, 2.0])
sweep = ParameterSweep(kappa_vals, sigma_vals, mu=100.0, t_max=0.5, n_steps=500)
half_lives = sweep.sweep_half_life()
assert half_lives.shape == (3, 2), '[TC08] 半衰期矩阵形状应为(3,2) FAILED'
assert abs(half_lives[0, 0] - np.log(2.0)) < 1e-10, '[TC08] κ=1时半衰期=ln2 FAILED'

# ---- TC09: 参数扫描 - 稳态方差矩阵形状与值 ----
var_infs = sweep.sweep_stationary_variance()
assert var_infs.shape == (3, 2), '[TC09] 稳态方差矩阵形状应为(3,2) FAILED'
assert abs(var_infs[0, 0] - 0.25 / 2.0) < 1e-10, '[TC09] σ=0.5,κ=1时稳态方差=σ²/2κ=0.125 FAILED'

# ---- TC10: 市场模拟 - VWAP计算 ----
np.random.seed(42)
sim = MarketSimulator(duration_seconds=5.0, base_price=100.0, seed=42)
events, sm = sim.run()
vwap = MarketSimulator.compute_vwap(events)
assert vwap is not None, '[TC10] 有带价格事件时VWAP不应为None FAILED'
assert 98.0 < vwap < 102.0, '[TC10] VWAP应在base_price附近 FAILED'

# ---- TC11: 市场模拟 - 订单不平衡度范围 ----
imbalance = MarketSimulator.compute_order_imbalance(events)
assert -1.0 <= imbalance <= 1.0, '[TC11] 订单不平衡度应在[-1,1]范围内 FAILED'

# ---- TC12: 订单簿 - 最优报价（逆向遍历返回最高买价和遇到的第一个卖价）----
lob = LimitOrderBook(base_price=100.0, tick_size=0.01, max_levels=50)
lob.add_order(100.05, 10, is_bid=True)
lob.add_order(99.95, 20, is_bid=True)
lob.add_order(100.10, 15, is_bid=False)
lob.add_order(100.20, 25, is_bid=False)
bb, ba, bbv, bav = lob.best_quotes()
assert bb is not None, '[TC12] 有买单时最优买价不应为None FAILED'
assert ba is not None, '[TC12] 有卖单时最优卖价不应为None FAILED'
assert bb > 100.0, '[TC12] 最优买价应大于base_price FAILED'

# ---- TC13: 订单簿 - 中间价与价差 ----
mid = lob.mid_price()
spread = lob.spread()
assert mid is not None, '[TC13] 有报价时中间价不应为None FAILED'
assert spread is not None, '[TC13] 有报价时价差不应为None FAILED'

# ---- TC14: 订单簿 - 市价单执行 ----
exec_vol, avg_price = lob.execute_market_order(5, is_buy=True)
assert exec_vol <= 25, '[TC14] 执行量不应超过可用深度 FAILED'

# ---- TC15: LOB几何分析 - 四边形面积 ----
quad = np.array([[0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 3.0, 3.0]])
area = LOBGeometryAnalyzer.quadrilateral_area(quad)
assert abs(area - 6.0) < 1e-10, '[TC15] 2×3矩形面积应为6 FAILED'

# ---- TC16: LOB几何分析 - 深度集中度 ----
depths = np.array([10, 20, 30, 40])
conc = LOBGeometryAnalyzer.depth_concentration(depths)
assert 0.0 < conc <= 1.0, '[TC16] 深度集中度应在(0,1]范围内 FAILED'

# ---- TC17: LOB插值 - 二次插值输出尺寸 ----
x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
y_nodes = np.array([1.0, 4.0, 9.0, 16.0])
x_query = np.linspace(0.0, 3.0, 10)
y_interp = LOBInterpolator.quadratic_interpolate(x_nodes, y_nodes, x_query)
assert len(y_interp) == 10, '[TC17] 插值输出长度应与查询点一致 FAILED'

# ---- TC18: 做市策略 - 报价函数 ----
strategy = MarketMakingStrategy(delta_bid=0.03, delta_ask=0.04)
bid, ask = strategy.quote(100.0)
assert bid == 99.97, '[TC18] 买价=mid-delta_bid FAILED'
assert ask == 100.04, '[TC18] 卖价=mid+delta_ask FAILED'

# ---- TC19: 做市策略 - 参数存取 ----
strategy.set_params(np.array([0.05, 0.06, 0.002]))
params = strategy.get_params()
assert abs(params[0] - 0.05) < 1e-10, '[TC19] delta_bid应正确存取 FAILED'
assert abs(params[1] - 0.06) < 1e-10, '[TC19] delta_ask应正确存取 FAILED'

# ---- TC20: 做市策略 - 成交与PnL计算 ----
np.random.seed(42)
strategy2 = MarketMakingStrategy(delta_bid=0.02, delta_ask=0.02, inventory_penalty=0.001)
strategy2.on_fill('buy', 100.0, 5)
assert strategy2.inventory == 5, '[TC20] 买入后库存应为5 FAILED'
strategy2.on_fill('sell', 101.0, 3)
assert strategy2.inventory == 2, '[TC20] 卖出后库存应为2 FAILED'
pnl = strategy2.mark_to_market_pnl(100.5)
assert isinstance(pnl, float), '[TC20] PnL应为浮点数 FAILED'

# ---- TC21: 协方差估计 - EWMA更新 ----
np.random.seed(42)
cov_est = CovarianceEstimator(n_assets=3, decay=0.94)
for _ in range(100):
    cov_est.update(np.random.normal(0, 0.1, 3))
corr = cov_est.get_correlation()
assert corr.shape == (3, 3), '[TC21] 相关矩阵形状应为(3,3) FAILED'
assert np.all(np.diag(corr) == 1.0), '[TC21] 相关矩阵对角线应为1 FAILED'

# ---- TC22: 共轭梯度法 - 对称正定系统求解 ----
from risk_engine import ConjugateGradientSolver
A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([5.0, 4.0])
cg = ConjugateGradientSolver(tol=1e-12)
x = cg.solve(A, b)
assert abs(x[0] - 1.0) < 1e-6, '[TC22] CG解x[0]应为1.0 FAILED'
assert abs(x[1] - 1.0) < 1e-6, '[TC22] CG解x[1]应为1.0 FAILED'

# ---- TC23: 最小方差组合 - 权重和为1 ----
np.random.seed(42)
cov_est2 = CovarianceEstimator(n_assets=4, decay=0.94)
for _ in range(200):
    cov_est2.update(np.random.normal(0.001, 0.05, 4))
mvp = MinimumVariancePortfolio(cov_est2)
w = mvp.solve()
assert abs(np.sum(w) - 1.0) < 1e-8, '[TC23] 最小方差组合权重和应为1 FAILED'
assert np.all(w >= -1e-10), '[TC23] 权重应非负 FAILED'

# ---- TC24: 风险指标 - VaR与ES ----
returns = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, -0.02, -0.04, 0.005])
var_95 = RiskMetrics.value_at_risk(returns, 0.9)
assert var_95 > 0, '[TC24] 历史VaR应为正值 FAILED'
es_95 = RiskMetrics.expected_shortfall(returns, 0.9)
assert es_95 >= var_95, '[TC24] ES应不小于VaR FAILED'

# ---- TC25: 风险指标 - 最大回撤 ----
cum = np.array([0.0, 0.1, 0.3, 0.2, 0.1, 0.4, 0.6])
mdd = RiskMetrics.max_drawdown(cum)
assert mdd >= 0.0, '[TC25] 最大回撤应非负 FAILED'

# ---- TC26: 特殊函数 - Ci在典型值处 ----
sf = SpecialFunctions()
ci_1 = sf.ci(1.0)
assert isinstance(ci_1, float), '[TC26] Ci(1)应为浮点数 FAILED'
assert np.isfinite(ci_1), '[TC26] Ci(1)应为有限值 FAILED'

# ---- TC27: 特殊函数 - Si在典型值处 ----
si_1 = sf.si(1.0)
assert isinstance(si_1, float), '[TC27] Si(1)应为浮点数 FAILED'
assert np.isfinite(si_1), '[TC27] Si(1)应为有限值 FAILED'
assert si_1 > 0.0, '[TC27] Si(1)应为正值 FAILED'

# ---- TC28: 特殊函数 - 正态CDF ----
ncdf_0 = sf.normal_cdf(0.0)
assert abs(ncdf_0 - 0.5) < 1e-6, '[TC28] N(0)应为0.5 FAILED'
ncdf_pos = sf.normal_cdf(3.0)
assert ncdf_pos > 0.99, '[TC28] N(3)应接近1 FAILED'
ncdf_neg = sf.normal_cdf(-3.0)
assert ncdf_neg < 0.01, '[TC28] N(-3)应接近0 FAILED'

# ---- TC29: 特殊函数 - Black-Scholes Delta ----
delta_call = sf.black_scholes_delta(S=100.0, K=100.0, T=0.25, r=0.05, sigma=0.2, option_type='call')
assert 0.45 < delta_call < 0.65, '[TC29] ATM Call Delta应在0.5附近 FAILED'
delta_put = sf.black_scholes_delta(S=100.0, K=100.0, T=0.25, r=0.05, sigma=0.2, option_type='put')
assert -0.55 < delta_put < -0.35, '[TC29] ATM Put Delta应在-0.5附近 FAILED'

# ---- TC30: 字典编码 - 构建字典与熵 ----
np.random.seed(42)
price_changes = np.random.uniform(-0.1, 0.1, 100)
volumes = np.random.randint(1, 20, 100)
sides = np.random.randint(0, 2, 100)
types = np.random.randint(0, 5, 100)
encoder = DictionaryEncoder(price_tick_size=0.01, volume_bucket_size=2)
encoded = encoder.build_dictionary(price_changes, volumes, sides, types)
assert len(encoded) == 100, '[TC30] 编码后长度应与输入一致 FAILED'
assert len(encoder.dictionary) > 0, '[TC30] 字典不应为空 FAILED'

# ---- TC31: 字典编码 - 游程编码 ----
rle = encoder.run_length_encode(encoded)
assert len(rle) > 0, '[TC31] 游程编码结果不应为空 FAILED'
assert sum(count for _, count in rle) == 100, '[TC31] 游程编码总计数应等于原长 FAILED'

# ---- TC32: 字典编码 - 信息熵 ----
entropy = encoder.compute_entropy()
assert entropy >= 0.0, '[TC32] 信息熵应非负 FAILED'

# ---- TC33: 分形维数 - 盒维数范围 ----
from chaos_analysis import FractalDimension
np.random.seed(42)
t_chaos = np.linspace(0, 1, 500)
y_chaos = np.cumsum(np.random.normal(0, 0.01, 500))
d_box = FractalDimension.box_counting_dimension(t_chaos, y_chaos)
assert 0.5 < d_box < 2.0, '[TC33] 价格路径盒维数应在(0.5,2.0)范围内 FAILED'

# ---- TC34: Hurst指数 - 白噪声约为0.5 ----
from chaos_analysis import HurstExponent
np.random.seed(42)
white_noise = np.random.normal(0, 1, 500)
H = HurstExponent.rescaled_range(white_noise)
assert 0.2 < H < 0.8, '[TC34] 白噪声Hurst指数应在0.5附近 FAILED'

# ---- TC35: Fekete积分 - ∫e^x dx = e-1/e ----
fq = FeketeQuadrature(-1.0, 1.0)
result = fq.integrate(lambda x: np.exp(x), m=10)
exact = np.exp(1.0) - np.exp(-1.0)
assert abs(result - exact) < 1e-8, '[TC35] Fekete积分误差应小于1e-8 FAILED'

# ---- TC36: Fekete积分 - ∫x² dx = 2/3 ----
result_x2 = fq.integrate(lambda x: x ** 2, m=10)
assert abs(result_x2 - 2.0 / 3.0) < 1e-6, '[TC36] ∫x² dz在[-1,1]上应为2/3 FAILED'

# ---- TC37: 多维积分 - ∫∫x²y²dxdy = 4/9 ----
result_2d = MultidimensionalQuadrature.tensor_product_2d(
    lambda x, y: x ** 2 * y ** 2, m1=8, m2=8, a1=-1.0, b1=1.0, a2=-1.0, b2=1.0)
assert abs(result_2d - 4.0 / 9.0) < 1e-6, '[TC37] 二维张量积积分应为4/9 FAILED'

# ---- TC38: Fekete点数量 ----
nf, xf, wf, vf = fq.compute_fekete_points(m=8, n_samples=200)
assert nf > 0, '[TC38] Fekete点数应大于0 FAILED'
assert len(xf) == nf, '[TC38] Fekete点坐标长度应等于nf FAILED'
assert len(wf) == nf, '[TC38] 权重长度应等于nf FAILED'

# ---- TC39: 多维积分 - 常数函数 ----
result_const = MultidimensionalQuadrature.tensor_product_2d(
    lambda x, y: 1.0, m1=6, m2=6, a1=0.0, b1=2.0, a2=0.0, b2=3.0)
assert abs(result_const - 6.0) < 1e-6, '[TC39] ∫∫1dxdy在[0,2]×[0,3]上应为6 FAILED'

# ---- TC40: 金融期望 - 非负payoff的期望非负 ----
payoff = lambda x: np.maximum(0.0, x)
exp_payoff = FinancialExpectation.expected_payoff_fekete(payoff, m=15, a=-5.0, b=5.0)
assert exp_payoff >= 0.0, '[TC40] max(0,X)的期望应非负 FAILED'

# ---- TC41: LOB几何 - market_thickness ----
thickness = LOBGeometryAnalyzer.market_thickness(0.05, 100.0)
assert thickness == 5.0, '[TC41] 市场厚度=价差×最大深度 FAILED'

# ---- TC42: 风险几何 - 凸包面积 ----
np.random.seed(42)
points = np.random.uniform(0, 1, (5, 2))
area_geo = RiskGeometry.convex_hull_area_2d(points)
assert area_geo >= 0.0, '[TC42] 凸包面积应非负 FAILED'

# ---- TC43: 风险几何 - 凸四边形判定 ----
square = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
assert RiskGeometry.is_convex_quadrilateral(square), '[TC43] 正方形应为凸四边形 FAILED'

# ---- TC44: 回测引擎 - 输出包含完整指标 ----
np.random.seed(42)
price_path = np.array([100.0 + 0.01 * i for i in range(100)])
time_grid = np.linspace(0.0, 1.0, 100)
engine = BacktestEngine(price_path=price_path, time_grid=time_grid, arrival_intensity=10.0, fill_probability_model='exponential')
strategy_test = MarketMakingStrategy(delta_bid=0.02, delta_ask=0.02)
metrics = engine.run(strategy_test, seed=42)
assert 'total_pnl' in metrics, '[TC44] 回测指标应包含total_pnl FAILED'
assert 'sharpe_ratio' in metrics, '[TC44] 回测指标应包含sharpe_ratio FAILED'
assert 'max_drawdown' in metrics, '[TC44] 回测指标应包含max_drawdown FAILED'

# ---- TC45: 回测引擎 - 可复现性 ----
np.random.seed(42)
strategy_a = MarketMakingStrategy(delta_bid=0.03, delta_ask=0.03)
metrics_a = engine.run(strategy_a, seed=123)
strategy_b = MarketMakingStrategy(delta_bid=0.03, delta_ask=0.03)
metrics_b = engine.run(strategy_b, seed=123)
assert abs(metrics_a['total_pnl'] - metrics_b['total_pnl']) < 1e-10, '[TC45] 相同参数和种子应产生相同回测结果 FAILED'

# ---- TC46: 做市策略 - 库存惩罚 ----
strategy_inv = MarketMakingStrategy(inventory_target=0, inventory_penalty=0.01)
strategy_inv.inventory = 10
penalty = strategy_inv.inventory_risk_penalty()
assert penalty == 1.0, '[TC46] 库存惩罚=η×(I-I*)² FAILED'

# ---- TC47: 协方差估计 - 协方差矩阵对称性 ----
np.random.seed(42)
cov_est3 = CovarianceEstimator(n_assets=3, decay=0.94)
for _ in range(50):
    cov_est3.update(np.random.normal(0, 0.05, 3))
cov_mat = cov_est3.get_covariance()
assert np.allclose(cov_mat, cov_mat.T), '[TC47] 协方差矩阵应对称 FAILED'

# ---- TC48: Calmar比率 ----
ret_simple = np.array([0.01, -0.005, 0.02, 0.01, -0.01])
cum_simple = np.cumsum(ret_simple)
calmar = RiskMetrics.calmar_ratio(ret_simple, cum_simple)
assert isinstance(calmar, float), '[TC48] Calmar比率应为浮点数 FAILED'

# ---- TC49: Cornish-Fisher VaR ----
np.random.seed(42)
ret_cf = np.random.normal(0, 0.02, 200)
cf_var = RiskMetrics.cornish_fisher_var(ret_cf, 0.95)
assert np.isfinite(cf_var), '[TC49] Cornish-Fisher VaR应为有限值 FAILED'

# ---- TC50: 综合回测流程 - 从OU模拟到策略评估 ----
np.random.seed(42)
ou_full = OrnsteinUhlenbeck(kappa=5.0, mu=100.0, sigma=1.0, s0=100.0, t_max=0.5, n_steps=200, seed=42)
t_full, price_full = ou_full.simulate_exact_milstein()
assert len(price_full) == 201, '[TC50] 模拟价格路径长度应为n_steps+1 FAILED'
tg = np.linspace(0.0, 0.5, len(price_full))
eng = BacktestEngine(price_path=price_full, time_grid=tg, arrival_intensity=20.0, fill_probability_model='exponential')
strat = MarketMakingStrategy(delta_bid=0.02, delta_ask=0.02)
met = eng.run(strat, seed=42)
assert met['final_inventory'] is not None, '[TC50] 回测应记录最终库存 FAILED'

print('\n全部 50 个测试通过!\n')
