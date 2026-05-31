# ---- TC01: results 是字典类型 ----
assert isinstance(results, dict), '[TC01] results 类型应为 dict FAILED'

# ---- TC02: data_shape 正确 ----
assert 'data_shape' in results, '[TC02] results 缺少 data_shape FAILED'
assert results['data_shape'] == (300, 20), '[TC02] data_shape 应为 (300,20) FAILED'

# ---- TC03: data_shape 为二维元组 ----
assert len(results['data_shape']) == 2, '[TC03] data_shape 应为二维 FAILED'

# ---- TC04: sampling 包含必要键 ----
sampling = results['sampling']
assert 'metric_tensor' in sampling, '[TC04] sampling 缺少 metric_tensor FAILED'
assert 'samples' in sampling, '[TC04] sampling 缺少 samples FAILED'
assert 'tangent_basis' in sampling, '[TC04] sampling 缺少 tangent_basis FAILED'
assert 'curvatures' in sampling, '[TC04] sampling 缺少 curvatures FAILED'

# ---- TC05: metric_tensor 为方阵 ----
mt = sampling['metric_tensor']
assert mt.shape[0] == mt.shape[1], '[TC05] metric_tensor 应为方阵 FAILED'
assert mt.shape[0] == 20, '[TC05] metric_tensor 维度应为20 FAILED'

# ---- TC06: curvatures 非负（容许数值误差） ----
curv = sampling['curvatures']
assert np.all(curv >= -1e-12), '[TC06] curvatures 应全部非负 FAILED'

# ---- TC07: graph 包含必要键 ----
graph = results['graph']
assert 'edges' in graph, '[TC07] graph 缺少 edges FAILED'
assert 'weights' in graph, '[TC07] graph 缺少 weights FAILED'
assert 'laplacian' in graph, '[TC07] graph 缺少 laplacian FAILED'

# ---- TC08: edges 形状正确 ----
edges = graph['edges']
assert edges.ndim == 2, '[TC08] edges 应为二维数组 FAILED'
assert edges.shape[1] == 2, '[TC08] edges 第二维应为2 FAILED'

# ---- TC09: weights 与 edges 长度一致 ----
weights = graph['weights']
assert len(weights) == len(edges), '[TC09] weights 与 edges 长度应一致 FAILED'

# ---- TC10: embedding 包含必要键 ----
embed = results['embedding']
assert 'embedding_linear' in embed, '[TC10] embedding 缺少 embedding_linear FAILED'
assert 'embedding_nonlinear' in embed, '[TC10] embedding 缺少 embedding_nonlinear FAILED'
assert 'eigenvalues' in embed, '[TC10] embedding 缺少 eigenvalues FAILED'
assert 'energy' in embed, '[TC10] embedding 缺少 energy FAILED'

# ---- TC11: NLSE 能量为有限值 ----
energy = embed['energy']
assert np.isfinite(energy), '[TC11] NLSE 能量应为有限值 FAILED'

# ---- TC12: embedding 形状正确 ----
emb_nl = embed['embedding_nonlinear']
assert emb_nl.shape[0] == 300, '[TC12] nonlinear embedding 行数应为300 FAILED'
assert emb_nl.shape[1] == 3, '[TC12] nonlinear embedding 列数应为3 FAILED'

# ---- TC13: eigenvalues 非负 ----
eigvals = embed['eigenvalues']
assert np.all(eigvals >= -1e-10), '[TC13] eigenvalues 应非负 FAILED'

# ---- TC14: spherical 包含必要键 ----
spherical = results['spherical']
assert 'spectrum' in spherical, '[TC14] spherical 缺少 spectrum FAILED'
assert 'coefficients' in spherical, '[TC14] spherical 缺少 coefficients FAILED'

# ---- TC15: spectrum 长度正确 ----
spectrum = spherical['spectrum']
assert len(spectrum) == 7, '[TC15] l_max=6 时 spectrum 长度应为7 FAILED'

# ---- TC16: spectrum 首项 ≥ 0 ----
assert spectrum[0] >= 0, '[TC16] spectrum[0] 应非负 FAILED'

# ---- TC17: quadrature 包含必要键 ----
quad = results['quadrature']
assert 'integral_1d' in quad, '[TC17] quadrature 缺少 integral_1d FAILED'
assert 'volume_elements' in quad, '[TC17] quadrature 缺少 volume_elements FAILED'

# ---- TC18: integral_1d 为正有限值 ----
assert quad['integral_1d'] > 0, '[TC18] integral_1d 应为正值 FAILED'
assert np.isfinite(quad['integral_1d']), '[TC18] integral_1d 应为有限值 FAILED'

# ---- TC19: volume_elements 非负 ----
vol = quad['volume_elements']
assert np.all(vol >= 0), '[TC19] volume_elements 应全部非负 FAILED'

# ---- TC20: gradient 包含必要键 ----
grad = results['gradient']
assert 'gradient' in grad, '[TC20] gradient 缺少 gradient FAILED'
assert 'selected_features' in grad, '[TC20] gradient 缺少 selected_features FAILED'
assert 'conserved_energy' in grad, '[TC20] gradient 缺少 conserved_energy FAILED'

# ---- TC21: conserved_energy 为有限值 ----
assert np.isfinite(grad['conserved_energy']), '[TC21] conserved_energy 应为有限值 FAILED'

# ---- TC22: topological 包含必要键 ----
topo = results['topological']
assert 'betti_0' in topo, '[TC22] topological 缺少 betti_0 FAILED'
assert 'persistence' in topo, '[TC22] topological 缺少 persistence FAILED'
assert 'topo_features' in topo, '[TC22] topological 缺少 topo_features FAILED'
assert 'lo_solution' in topo, '[TC22] topological 缺少 lo_solution FAILED'

# ---- TC23: betti_0 ≥ 1 ----
assert topo['betti_0'] >= 1, '[TC23] betti_0 应 ≥ 1 FAILED'

# ---- TC24: lo_solution 为二元数组 ----
lo_sol = topo['lo_solution']
assert np.all((lo_sol == 0) | (lo_sol == 1)), '[TC24] lo_solution 元素应为 0 或 1 FAILED'

# ---- TC25: curve 包含必要键 ----
curve = results['curve']
assert 'arc_length' in curve, '[TC25] curve 缺少 arc_length FAILED'
assert 'curvature_max' in curve, '[TC25] curve 缺少 curvature_max FAILED'
assert 'geodesic_ratio' in curve, '[TC25] curve 缺少 geodesic_ratio FAILED'

# ---- TC26: arc_length > 0 ----
assert curve['arc_length'] > 0, '[TC26] arc_length 应为正值 FAILED'

# ---- TC27: curvature_max > 0 ----
assert curve['curvature_max'] > 0, '[TC27] curvature_max 应为正值 FAILED'

# ---- TC28: piecewise 包含必要键 ----
pwc = results['piecewise']
assert 'density' in pwc, '[TC28] piecewise 缺少 density FAILED'
assert 'entropy' in pwc, '[TC28] piecewise 缺少 entropy FAILED'
assert 'mutual_info' in pwc, '[TC28] piecewise 缺少 mutual_info FAILED'

# ---- TC29: entropy 为有限值 ----
assert np.isfinite(pwc['entropy']), '[TC29] entropy 应为有限值 FAILED'

# ---- TC30: mutual_info ≥ 0 ----
assert pwc['mutual_info'] >= 0, '[TC30] mutual_info 应 ≥ 0 FAILED'

# ---- TC31: algebra 包含必要键 ----
alg = results['algebra']
assert 'binary_data' in alg, '[TC31] algebra 缺少 binary_data FAILED'
assert 'hash_codes' in alg, '[TC31] algebra 缺少 hash_codes FAILED'
assert 'lo_features' in alg, '[TC31] algebra 缺少 lo_features FAILED'

# ---- TC32: binary_data 元素为 0/1 ----
bin_data = alg['binary_data']
assert np.all((bin_data == 0) | (bin_data == 1)), '[TC32] binary_data 元素应为 0 或 1 FAILED'

# ---- TC33: hash_codes 形状正确 ----
hc = alg['hash_codes']
assert hc.shape[0] == 300, '[TC33] hash_codes 行数应为300 FAILED'
assert hc.shape[1] == 16, '[TC33] hash_codes 列数应为16 FAILED'

# ---- TC34: metrics 包含必要键 ----
met = results['metrics']
assert 'isometric_quality' in met, '[TC34] metrics 缺少 isometric_quality FAILED'
assert 'trustworthiness' in met, '[TC34] metrics 缺少 trustworthiness FAILED'
assert 'reconstruction_error' in met, '[TC34] metrics 缺少 reconstruction_error FAILED'

# ---- TC35: isometric_quality 在 [0, 1] 内 ----
assert 0.0 <= met['isometric_quality'] <= 1.0, '[TC35] isometric_quality 应在 [0,1] FAILED'

# ---- TC36: trustworthiness 在 [0, 1] 内 ----
assert 0.0 <= met['trustworthiness'] <= 1.0, '[TC36] trustworthiness 应在 [0,1] FAILED'

# ---- TC37: elapsed_time > 0 ----
assert results['elapsed_time'] > 0, '[TC37] elapsed_time 应为正值 FAILED'

# ---- TC38: 可复现性——固定种子两次结果一致 ----
np.random.seed(42)
data1 = generate_synthetic_manifold_data(n_points=100, ambient_dim=10, intrinsic_dim=2)
np.random.seed(42)
data2 = generate_synthetic_manifold_data(n_points=100, ambient_dim=10, intrinsic_dim=2)
assert np.allclose(data1, data2), '[TC38] 固定种子两次生成数据应一致 FAILED'

# ---- TC39: generate_synthetic_manifold_data 输出尺寸正确 ----
np.random.seed(42)
data_test = generate_synthetic_manifold_data(n_points=50, ambient_dim=8, intrinsic_dim=2)
assert data_test.shape == (50, 8), '[TC39] 生成数据形状应为 (50,8) FAILED'
assert np.all(np.isfinite(data_test)), '[TC39] 生成数据应全部有限 FAILED'

# ---- TC40: chord_length > 0 ----
assert curve['arc_length'] > 0, '[TC40] arc_length 应为正值 FAILED'

# ---- TC41: lo_solution 维度为25 ----
assert len(lo_sol) == 25, '[TC41] lo_solution 长度应为25 (5×5) FAILED'

# ---- TC42: spectrum 能量总和 ≥ 0 ----
assert np.sum(spectrum) >= 0, '[TC42] spectrum 能量总和应 ≥ 0 FAILED'

# ---- TC43: persistence 字典非空 ----
persistence = topo['persistence']
assert len(persistence) > 0, '[TC43] persistence 字典不应为空 FAILED'

# ---- TC44: reconstruction_error ≥ 0 ----
assert met['reconstruction_error'] >= 0, '[TC44] reconstruction_error 应 ≥ 0 FAILED'

# ---- TC45: hash_codes 元素为 0/1 ----
assert np.all((hc == 0) | (hc == 1)), '[TC45] hash_codes 元素应为 0 或 1 FAILED'
