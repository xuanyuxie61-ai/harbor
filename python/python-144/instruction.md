# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# 项目描述：高维风险平价投资组合优化系统

本项目是一个金融工程综合计算框架，目标是对投资组合进行高精度风险度量、网络风险传播分析、蒙特卡洛模拟、投资组合优化（含风险平价与最小方差）、以及耦合动力学模拟。主程序 `main.py` 调用了一系列专用模块，利用谱方法、随机微分方程、图论与优化技术，完成从数据生成到最终策略评估的全流程。

## 被移除的文件清单及职责

需要根据 `main.py` 中的使用方式实现以下七个 Python 模块，每个模块封装一组相关算法。

### 1. `chebyshev_pricing.py` – Chebyshev 谱方法风险计算
- **`chebyshev_grid(n)`**：生成 n+1 个 Chebyshev 节点 (cos 形式)，用于在 [-1,1] 上构建插值。
- **`chebyshev_diff_matrix(n)`**：返回 Chebyshev 谱微分矩阵 D，使得 w = D·v 近似导数。
- **`chebyshev_barycentric_interpolate(x_grid, v, x_query)`**：用重心 Lagrange 插值在 Chebyshev 节点上计算查询点的函数值。
- **`spectral_var_cvar(returns, alpha, n_cheb)`**：核心风险度量函数。将收益率样本映射到标准区间，通过 Chebyshev 节点上的经验 CDF 插值及谱微分计算 PDF，然后解 CDF=α 的根得 VaR，再在尾部积分得到 CVaR。返回包含 VaR、CVaR、均值、标准差及谱节点信息的字典。
- **`circle01_monomial_integral(e)`**：计算单位圆周上单项式解析积分，用于验证。

### 2. `dynamics_model.py` – 耦合市场动力学与 SDE 求解
- **`coupled_market_dynamics(y, t, k1, K2, gamma, m)`**：计算高维耦合弹簧系统的右端项。状态向量 y 为 [u₁,v₁,u₂,v₂,…]，结合个体均值回归、资产间耦合与阻尼，返回 dy/dt。
- **`trapezoidal_sde_solver(f, g, tspan, y0, n_steps, rng)`**：梯形隐式格式求解 SDE，对漂移项采用 Crank‑Nicolson 方式隐式迭代，对扩散项采用显式 Euler‑Maruyama，每步使用不动点迭代解非线性方程。
- **`trapezoidal_ode_solver(f, tspan, y0, n_steps)`**：确定性梯形法 ODE 求解器，类似但无随机项。
- **`simulate_contagion(n_assets, T, dt, k1, gamma, sigma_noise, rng)`**：模拟资产传染场景，构建星形耦合矩阵，并在时间中点施加冲击，返回轨迹和最大偏离。

### 3. `monte_carlo_simulator.py` – 蒙特卡洛与 Bootstrap
- **`simulate_returns_mc(mu, sigma, corr, T, n_paths, rng)`**：使用几何布朗运动和相关结构生成多维收益率路径，通过 Cholesky 分解耦合噪声。
- **`bootstrap_risk_analysis(returns, n_bootstrap, alpha, rng)`**：对等权重组合收益率进行 Bootstrap 重采样，计算 VaR、CVaR 的均值与置信区间。
- **`tournament_risk_simulation(strengths, n_games, rng)`**：模拟锦标赛，按资产强度随机决定每轮获胜者，返回胜率作为相对表现概率。
- **`high_dim_sphere_sampling(n_samples, dim, rng)`**：在高维单位球面上均匀采样（正态化方法）。

### 4. `network_risk.py` – 资产网络与风险传播
- **`build_asset_digraph(n, threshold, corr)`**：根据相关性阈值构建有向图邻接矩阵，孤立节点添加自环。
- **`pagerank_systemic_risk(adj, damping, max_iter, tol)`**：基于行随机矩阵的幂迭代计算 PageRank 得分，衡量系统重要性。
- **`delaunay_similarity_triangulation(positions)`**：对二维嵌入点进行 Delaunay 三角剖分，返回无向邻接矩阵。
- **`stochastic_risk_diffusion(network_adj, initial_risk, omega, nx, ny)`**：在二维网格上用五点差分法求解稳态随机热方程，系数受随机参数 ω 控制，边界固定为 ω₂，源项来自 `initial_risk`。
- **`network_risk_contribution(adj, asset_returns)`**：计算各资产的网络风险贡献度，基于波动率与加权 β 系数的乘积。

### 5. `portfolio_optimizer.py` – 投资组合优化
- **`markowitz_min_variance(Sigma, target_return, mu)`**：求解带非负约束的最小方差组合（可附加目标收益约束），使用 SLSQP 优化器。
- **`risk_parity_weights(Sigma, risk_budget, max_iter, tol)`**：用循环坐标下降（CCD）迭代求解风险平价权重，使各资产风险贡献与预算成正比，返回权重、风险贡献、组合风险、分散化比率等。
- **`herfindahl_risk_concentration(rc)`**：风险贡献的 Herfindahl 指数。
- **`effective_number_of_bets(rc)`**：基于熵的有效赌注数。
- **`risk_parity_with_budget_constraints(Sigma, lower, upper, risk_budget, max_iter)`**：带上下界约束的风险平价，通过投影梯度法优化凸目标。

### 6. `simplex_search.py` – 单纯形格点搜索
- **`simplex_lattice_points(n, t)`**：按逆字典序枚举 n 维标准单纯形上所有整数格点（总和为 t）。
- **`simplex_volume(points)`**：计算单纯形体积。
- **`covariance_simplex_volume(Sigma)`**：通过 Cholesky 分解计算协方差矩阵对应的广义体积（det(L)）。
- **`tet_quality_indicator_from_cov(Sigma)`**：从协方差子矩阵计算条件数等质量指标。
- **`lattice_portfolio_search(n_assets, t, Sigma, mu)`**：枚举所有格点权重并评估风险（及夏普比率），返回最优组合。
- **`mesh_base_one(element_node, node_num)`**：索引基准修正（0‑based 转 1‑based 或识别）。

### 7. `spherical_embedding.py` – 球面嵌入与分散度
- **`sphere_distance1(lat1, lon1, lat2, lon2, r)`**：Haversine 大圆距离。
- **`ll_to_xyz(r, ll)`** 与 **`xyz_to_ll(xyz, r)`**：经纬度 ↔ 笛卡尔坐标互换。
- **`map_spherical_residual(ll_vec, r, city_num, distance)`**：球面嵌入的残差函数，固定前两个点的自由度，输出距离差异。
- **`correlation_to_spherical_embedding(corr, r, random_seed)`**：通过非线性最小二乘将相关性矩阵映射为球面上点的三维笛卡尔坐标。
- **`circle01_sample_random(n, rng)`**：在单位圆上均匀采样。
- **`spherical_diversity_index(xyz)`**：基于球面点集重心范数计算分散度指标。
- **`angular_distance_matrix(xyz)`**：点集间的角度距离矩阵。

### 8. `utils.py` – 通用工具
- **`caesar_perturb(data, k, axis)`**：循环移位并加高斯噪声的扰动。
- **`matrix_interpolation_upsample(A, factor)`**：矩阵双线性上采样（factor=2）。
- **`polygonal_convex_hull(points)`**：计算点集凸包的顶点、体积等信息。
- **`distance_to_position_mds(distance, dim)`**：经典
