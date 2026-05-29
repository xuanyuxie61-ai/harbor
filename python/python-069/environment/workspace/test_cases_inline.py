import utils

# ---- TC01: utils.safe_divide 正常除法返回正确值 ----
assert abs(utils.safe_divide(10.0, 2.0) - 5.0) < 1e-12, '[TC01] utils.safe_divide 正常除法返回正确值 FAILED'

# ---- TC02: utils.safe_divide 零除法返回 fill_value ----
assert abs(utils.safe_divide(10.0, 0.0) - 0.0) < 1e-12, '[TC02] utils.safe_divide 零除法返回 fill_value FAILED'

# ---- TC03: utils.triangle_area_2d 直角三角形面积等于0.5 ----
t3 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
assert abs(utils.triangle_area_2d(t3) - 0.5) < 1e-12, '[TC03] utils.triangle_area_2d 直角三角形面积等于0.5 FAILED'

# ---- TC04: canopy_geometry.canopy_lai_profile 边界处返回0 ----
assert canopy_geometry.canopy_lai_profile(0.0, 20.0, 4.5) == 0.0, '[TC04] canopy_geometry.canopy_lai_profile 边界处返回0 FAILED'

# ---- TC05: canopy_geometry.canopy_lai_profile 冠层中部正值 ----
assert canopy_geometry.canopy_lai_profile(10.0, 20.0, 4.5) > 0.0, '[TC05] canopy_geometry.canopy_lai_profile 冠层中部正值 FAILED'

# ---- TC06: radiation_transfer.beer_lambert_irradiance 零LAI时光强不变 ----
assert abs(radiation_transfer.beer_lambert_irradiance(1200.0, 0.5, 0.0) - 1200.0) < 1e-9, '[TC06] radiation_transfer.beer_lambert_irradiance 零LAI时光强不变 FAILED'

# ---- TC07: radiation_transfer.beer_lambert_irradiance 正LAI时光强衰减 ----
I_bl = radiation_transfer.beer_lambert_irradiance(1200.0, 0.5, 2.0)
assert I_bl < 1200.0 and I_bl > 0.0, '[TC07] radiation_transfer.beer_lambert_irradiance 正LAI时光强衰减 FAILED'

# ---- TC08: radiation_transfer.campbell_ellipsoid_g 天顶角0时G等于0.5 ----
assert abs(radiation_transfer.campbell_ellipsoid_g(0.0) - 0.5) < 1e-12, '[TC08] radiation_transfer.campbell_ellipsoid_g 天顶角0时G等于0.5 FAILED'

# ---- TC09: photosynthesis_model.farquhar_photosynthesis I为0时J等于0 ----
an_tc, wc_tc, wj_tc, j_tc, rd_tc = photosynthesis_model.farquhar_photosynthesis(280.0, 210.0, 298.15, 0.0)
assert abs(j_tc) < 1e-9, '[TC09] photosynthesis_model.farquhar_photosynthesis I为0时J等于0 FAILED'

# ---- TC10: photosynthesis_model.temperature_sensitivity_centered 输出为有限值 ----
d_adt = photosynthesis_model.temperature_sensitivity_centered(280.0, 210.0, 298.15, 800.0, h=0.5)
assert np.isfinite(d_adt), '[TC10] photosynthesis_model.temperature_sensitivity_centered 输出为有限值 FAILED'

# ---- TC11: soil_carbon_flux.lloyd_taylor_soil_respiration 输出非负 ----
assert soil_carbon_flux.lloyd_taylor_soil_respiration(18.0) >= 0.0, '[TC11] soil_carbon_flux.lloyd_taylor_soil_respiration 输出非负 FAILED'

# ---- TC12: soil_carbon_flux.compute_nee 解析验证 ----
nee_tc = soil_carbon_flux.compute_nee(100.0, 20.0, 10.0)
assert abs(nee_tc - (-70.0)) < 1e-9, '[TC12] soil_carbon_flux.compute_nee 解析验证 FAILED'

# ---- TC13: carbon_allocation.greedy_carbon_allocation 总分配不超过总碳 ----
demands_tc = {'leaf': 40.0, 'stem': 30.0, 'root': 20.0, 'storage': 20.0}
biomass_tc = {'leaf': 500.0, 'stem': 2000.0, 'root': 800.0}
costs_tc = {'leaf': 1.5, 'stem': 2.0, 'root': 1.8, 'storage': 1.0}
allocated_tc = carbon_allocation.greedy_carbon_allocation(100.0, demands_tc, biomass_tc, costs_tc)
assert sum(allocated_tc.values()) <= 100.0 + 1e-9, '[TC13] carbon_allocation.greedy_carbon_allocation 总分配不超过总碳 FAILED'

# ---- TC14: carbon_allocation.compute_allocation_efficiency 范围在0到1之间 ----
eff_tc = carbon_allocation.compute_allocation_efficiency(allocated_tc, demands_tc)
assert 0.0 <= eff_tc <= 1.0, '[TC14] carbon_allocation.compute_allocation_efficiency 范围在0到1之间 FAILED'

# ---- TC15: environment_response.temperature_response 最适温度返回1 ----
assert abs(environment_response.temperature_response(25.0) - 1.0) < 1e-12, '[TC15] environment_response.temperature_response 最适温度返回1 FAILED'

# ---- TC16: environment_response.vpd_response 零VPD返回1 ----
assert abs(environment_response.vpd_response(0.0) - 1.0) < 1e-12, '[TC16] environment_response.vpd_response 零VPD返回1 FAILED'

# ---- TC17: environment_response.piecewise_linear_interp 中点插值准确性 ----
interp_val = environment_response.piecewise_linear_interp([0.0, 10.0, 20.0], [0.0, 0.5, 1.0], 15.0)
assert abs(interp_val - 0.75) < 1e-12, '[TC17] environment_response.piecewise_linear_interp 中点插值准确性 FAILED'

# ---- TC18: boundary_flux.hexagon01_area 解析验证 ----
assert abs(boundary_flux.hexagon01_area() - 3.0 * np.sqrt(3.0) / 2.0) < 1e-12, '[TC18] boundary_flux.hexagon01_area 解析验证 FAILED'

# ---- TC19: data_assimilation.simple_kalman_update 方差减小 ----
x_a_tc, P_a_tc = data_assimilation.simple_kalman_update(5.0, 4.0, 6.0, 1.0, 2.0)
assert P_a_tc <= 4.0 + 1e-12, '[TC19] data_assimilation.simple_kalman_update 方差减小 FAILED'

# ---- TC20: leaf_angle_sampling.simplex_unit_volume 3维等于1除以6 ----
assert abs(leaf_angle_sampling.simplex_unit_volume(3) - 1.0 / 6.0) < 1e-12, '[TC20] leaf_angle_sampling.simplex_unit_volume 3维等于1除以6 FAILED'

# ---- TC21: co2_diffusion.build_laplacian_2d 矩阵形状正确 ----
L_tc = co2_diffusion.build_laplacian_2d(4, 1.0)
assert L_tc.shape == (16, 16), '[TC21] co2_diffusion.build_laplacian_2d 矩阵形状正确 FAILED'

# ---- TC22: microclimate_fem.tetens_vapor_pressure 输出非负 ----
assert microclimate_fem.tetens_vapor_pressure(25.0) >= 0.0, '[TC22] microclimate_fem.tetens_vapor_pressure 输出非负 FAILED'

# ---- TC23: optimization_parser.solve_carbon_lp 总消耗不超过总碳 ----
coeffs_tc = {'leaf': 1.2, 'stem': 1.0, 'root': 0.9, 'storage': 0.7}
maint_tc = {'leaf': 10.0, 'stem': 5.0, 'root': 5.0, 'storage': 0.0}
sol_lp_tc, obj_lp_tc = optimization_parser.solve_carbon_lp(100.0, coeffs_tc, maint_tc)
assert sum(sol_lp_tc.values()) <= 100.0 + 1e-9, '[TC23] optimization_parser.solve_carbon_lp 总消耗不超过总碳 FAILED'

# ---- TC24: cvt_leaf_distribution.canopy_cvt_optimization 能量单调递减 ----
np.random.seed(42)
g_opt_tc, e_hist_tc, m_hist_tc = cvt_leaf_distribution.canopy_cvt_optimization(20.0, 5.0, n_clusters=10, it_num=5, s_num=10)
assert e_hist_tc[-1] <= e_hist_tc[0], '[TC24] cvt_leaf_distribution.canopy_cvt_optimization 能量单调递减 FAILED'

# ---- TC25: main.main 集成测试返回结构正确 ----
np.random.seed(42)
result_tc = main()
assert isinstance(result_tc, dict) and 'an' in result_tc and 'nee' in result_tc and 'allocation' in result_tc, '[TC25] main.main 集成测试返回结构正确 FAILED'
