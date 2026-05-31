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
