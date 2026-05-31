# ---- TC01: extract_room_surfaces 返回 6 个表面 ----
test_surfaces = extract_room_surfaces()
assert len(test_surfaces) == 6, '[TC01] surface count FAILED'
assert set(test_surfaces.keys()) == {'floor', 'ceiling', 'front_wall', 'back_wall', 'left_wall', 'right_wall'}, '[TC01] surface names FAILED'

# ---- TC02: room_surface_areas 地板面积等于 80 m² ----
test_areas = room_surface_areas(test_surfaces)
assert abs(test_areas['floor'] - 80.0) < 1e-10, '[TC02] floor area FAILED'
assert abs(test_areas['ceiling'] - 80.0) < 1e-10, '[TC02] ceiling area FAILED'

# ---- TC03: room_total_volume 接近 400 减去柱子体积 ----
test_vol = room_total_volume()
expected_vol = 400.0 - 2.0 * (np.pi * 0.3 ** 2 * 5.0)
assert abs(test_vol - expected_vol) < 1e-10, '[TC03] room volume FAILED'

# ---- TC04: compute_sabine_reverberation_time 返回正值 ----
test_abs = {'floor': 0.15, 'ceiling': 0.25, 'front_wall': 0.10, 'back_wall': 0.10, 'left_wall': 0.05, 'right_wall': 0.05}
test_t60 = compute_sabine_reverberation_time(test_abs, test_surfaces)
assert test_t60 > 0.5 and test_t60 < 5.0, '[TC04] Sabine T60 range FAILED'

# ---- TC05: mesh_statistics 返回正节点数和四面体数 ----
np.random.seed(42)
h0 = 1.2
box = [0.0, 10.0, 0.0, 8.0, 0.0, 5.0]
pfix = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 8.0, 0.0], [0.0, 8.0, 0.0],
                 [0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [10.0, 8.0, 5.0], [0.0, 8.0, 5.0]], dtype=float)
p, t = distmesh_3d(dshoebox_with_pillars, huniform, h0, box, iteration_max=10, pfix=pfix)
stats = mesh_statistics(p, t)
assert stats['node_num'] > 0, '[TC05] node count FAILED'
assert stats['tet_num'] > 0, '[TC05] tet count FAILED'
assert stats['volume_min'] > 0, '[TC05] volume min FAILED'

# ---- TC06: simp_qual_3d 四面体质量在 [0, 100] 内 ----
qual = simp_qual_3d(p, t)
assert np.all(qual >= 0.0) and np.all(qual <= 100.0), '[TC06] quality range FAILED'

# ---- TC07: surftri 提取表面三角形数量大于 0 ----
boundary_faces = surftri(p, t)
assert boundary_faces.shape[0] > 0, '[TC07] boundary faces count FAILED'
assert boundary_faces.shape[1] == 3, '[TC07] boundary faces dim FAILED'

# ---- TC08: compute_sound_pressure_level 标量输入输出正 dB ----
spl_scalar = compute_sound_pressure_level(np.array([1.0]))
assert spl_scalar > 0, '[TC08] SPL positive FAILED'

# ---- TC09: SparseCOO 矩阵向量乘法正确性 ----
A_coo = SparseCOO(np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]), (3, 3))
x_vec = np.array([1.0, 1.0, 1.0])
y_vec = A_coo.mv(x_vec)
assert np.allclose(y_vec, np.array([1.0, 2.0, 3.0])), '[TC09] SparseCOO mv FAILED'

# ---- TC10: conjugate_gradient 求解单位矩阵方程 ----
I_coo = SparseCOO(np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1.0, 1.0, 1.0]), (3, 3))
b_vec = np.array([3.0, -1.0, 2.0])
x_sol = conjugate_gradient(I_coo, b_vec, tol=1e-10)
assert np.allclose(x_sol, b_vec, atol=1e-8), '[TC10] CG solve identity FAILED'

# ---- TC11: rectangular_room_modes 基频等于 c/(2*Lx) ----
modes = rectangular_room_modes(10.0, 8.0, 5.0, max_order=1)
base_freq = modes[0]['frequency']
expected_base = C_AIR / (2.0 * 10.0)
assert abs(base_freq - expected_base) < 1e-10, '[TC11] base mode frequency FAILED'

# ---- TC12: schroeder_frequency 返回值正有限 ----
fs = schroeder_frequency(400.0, 340.0, 0.1)
assert np.isfinite(fs) and fs > 0, '[TC12] Schroeder frequency finite FAILED'

# ---- TC13: compute_modal_density 与 f² 成正比 ----
nf1 = compute_modal_density(400.0, 100.0)
nf2 = compute_modal_density(400.0, 200.0)
assert abs(nf2 / nf1 - 4.0) < 1e-10, '[TC13] modal density scaling FAILED'

# ---- TC14: ball01_monomial_integral 奇数指数返回 0 ----
val_odd = ball01_monomial_integral((1, 0, 0))
assert abs(val_odd) < 1e-14, '[TC14] ball odd exponent FAILED'

# ---- TC15: ball01_monomial_integral (0,0,0) 等于单位球体积 ----
val_000 = ball01_monomial_integral((0, 0, 0))
assert abs(val_000 - 4.0 * np.pi / 3.0) < 1e-10, '[TC15] ball zero exponent FAILED'

# ---- TC16: integrate_over_triangle 常数函数积分等于面积 ----
v0 = np.array([0.0, 0.0, 0.0])
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
I_const = integrate_over_triangle(lambda p: 1.0, v0, v1, v2, precision=5)
assert abs(I_const - 0.5) < 1e-10, '[TC16] integrate constant FAILED'

# ---- TC17: detect_room_edges 返回 12 条边 ----
test_edges = detect_room_edges(test_surfaces)
assert len(test_edges) == 12, '[TC17] room edges count FAILED'

# ---- TC18: compute_mean_free_path 理论值验证 ----
mfp = compute_mean_free_path(400.0, 340.0)
assert abs(mfp - 4.0 * 400.0 / 340.0) < 1e-10, '[TC18] mean free path FAILED'

# ---- TC19: linear_regression 斜率与截距精确恢复 ----
x_lr = np.array([0.0, 1.0, 2.0, 3.0])
y_lr = np.array([1.0, 3.0, 5.0, 7.0])
slope_lr, int_lr = linear_regression(x_lr, y_lr)
assert abs(slope_lr - 2.0) < 1e-10 and abs(int_lr - 1.0) < 1e-10, '[TC19] linear regression FAILED'

# ---- TC20: eyring_absorption_to_t60 平均吸声系数为 0 时退化为 Sabine ----
t60_eyring_zero = eyring_absorption_to_t60(400.0, 340.0, 0.0)
t60_sabine_zero = sabine_absorption_to_t60(400.0, 340.0 * 0.0)
assert abs(t60_eyring_zero - t60_sabine_zero) < 1e-6, '[TC20] Eyring zero absorption FAILED'

# ---- TC21: C_AIR 声速常数等于 343.0 ----
assert C_AIR == 343.0, '[TC21] C_AIR constant FAILED'

# ---- TC22: compute_statistics 返回字典包含均值和标准差 ----
stats_dict = compute_statistics(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
assert 'mean' in stats_dict and 'std' in stats_dict, '[TC22] statistics keys FAILED'
assert abs(stats_dict['mean'] - 3.0) < 1e-10, '[TC22] statistics mean FAILED'

# ---- TC23: modal_overlap_factor 输出列表长度与输入一致 ----
mof_data = modal_overlap_factor(modes[:5], damping_ratio=0.01)
assert len(mof_data) == 5, '[TC23] MOF list length FAILED'
assert all('mof' in d and 'frequency' in d for d in mof_data), '[TC23] MOF keys FAILED'

# ---- TC24: ball01_sample 所有采样点位于单位球内 ----
np.random.seed(42)
samples = ball01_sample(100)
norm_samples = np.linalg.norm(samples, axis=1)
assert np.all(norm_samples <= 1.0 + 1e-10), '[TC24] ball sample inside FAILED'

# ---- TC25: monte_carlo_ray_tracing 返回非负 T60 ----
np.random.seed(42)
normals = compute_surface_normals(test_surfaces)
times, edc, T60_mc, EDT = monte_carlo_ray_tracing(test_surfaces, normals, test_abs,
                                                   np.array([5.0, 4.0, 2.5]),
                                                   n_rays=100, max_reflections=20)
assert T60_mc >= 0.0, '[TC25] MC T60 non-negative FAILED'
assert EDT >= 0.0, '[TC26] EDT non-negative FAILED'

# ---- TC26: build_reflection_graph 转移概率矩阵每行和为 1 ----
np.random.seed(42)
trans_prob, surf_names = build_reflection_graph(test_surfaces, normals, test_abs, n_rays=200)
row_sums = trans_prob.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), '[TC26] transition row sums FAILED'

# ---- TC27: compute_room_response_stats 返回结构正确 ----
np.random.seed(42)
rt_stats = compute_room_response_stats(test_surfaces, normals, test_abs,
                                        np.array([5.0, 4.0, 2.5]), n_rays=100)
assert 'mean_free_path' in rt_stats, '[TC27] response stats keys FAILED'
assert rt_stats['mean_free_path'] >= 0.0, '[TC27] mean free path non-negative FAILED'

# ---- TC28: compute_edge_diffraction_field 返回有限复数值 ----
receiver_pos = np.array([7.0, 6.0, 3.0])
diff_field = compute_edge_diffraction_field(np.array([5.0, 4.0, 2.5]), receiver_pos, test_edges, 500.0)
assert np.isfinite(np.abs(diff_field)), '[TC28] diffraction field finite FAILED'

# ---- TC29: solve_helmholtz_cg 返回解向量长度等于节点数 ----
np.random.seed(42)
p_small = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
t_small = np.array([[0, 1, 2, 3]], dtype=int)
p_sol_test, k_wavenum_test = solve_helmholtz_cg(p_small, t_small, 125.0, source_node=0, source_strength=1.0, tol=1e-6)
assert len(p_sol_test) == p_small.shape[0], '[TC29] Helmholtz solution length FAILED'
assert k_wavenum_test > 0, '[TC29] wavenumber positive FAILED'

# ---- TC30: triangle_symq_rule 精度 1 返回 1 个求积点 ----
n_rule, a_rule, b_rule, c_rule, w_rule = triangle_symq_rule(1)
assert n_rule == 1, '[TC30] triangle rule 1 point FAILED'
assert abs(a_rule[0] - 1.0/3.0) < 1e-10, '[TC30] triangle rule barycenter FAILED'
