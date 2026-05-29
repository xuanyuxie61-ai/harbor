# ---- TC01: spherical_volume 正常计算球壳体积 ----
from utils import spherical_volume
vol = spherical_volume(0.0, 1.0)
assert abs(vol - 4.0 * np.pi / 3.0) < 1.0e-10, '[TC01] spherical_volume 正常计算 FAILED'

# ---- TC02: spherical_volume 无效输入返回0 ----
vol_invalid = spherical_volume(1.0, 0.5)
assert vol_invalid == 0.0, '[TC02] spherical_volume 无效输入 FAILED'

# ---- TC03: vector_norm 计算正确 ----
from utils import vector_norm
v = np.array([3.0, 4.0, 0.0])
assert abs(vector_norm(v) - 5.0) < 1.0e-10, '[TC03] vector_norm FAILED'

# ---- TC04: normalize_vector 返回单位向量 ----
from utils import normalize_vector
v2 = np.array([1.0, 2.0, 2.0])
nv = normalize_vector(v2)
assert abs(vector_norm(nv) - 1.0) < 1.0e-10, '[TC04] normalize_vector FAILED'

# ---- TC05: log_mean 对数平均正确性 ----
from utils import log_mean
lm = log_mean(1.0, np.e)
assert abs(lm - (np.e - 1.0)) < 1.0e-10, '[TC05] log_mean FAILED'

# ---- TC06: electron_number_density 标量输入返回正值 ----
n_e = electron_number_density(1000.0, 1.0, 2.5)
assert n_e > 0.0 and np.isfinite(n_e), '[TC06] electron_number_density FAILED'

# ---- TC07: ionization_state_Saha 低温返回接近0 ----
Z_saha = ionization_state_Saha(100.0, 10.0, 1.0, 13.6 * PC.ELEMENTARY_CHARGE)
assert Z_saha < 0.01, '[TC07] ionization_state_Saha 低温 FAILED'

# ---- TC08: RadialMesh 初始化节点数正确 ----
mesh = RadialMesh(n_cells=50)
assert mesh.n_nodes >= 51, '[TC08] RadialMesh 节点数 FAILED'

# ---- TC09: RadialMesh.cell_volumes 全为正 ----
vols = mesh.cell_volumes()
assert np.sum(vols) > 0.0, '[TC09] cell_volumes 总和为正 FAILED'

# ---- TC10: build_1d_fem_adjacency 返回正确结构 ----
adj = build_1d_fem_adjacency(5)
assert len(adj) == 5 and adj[0] == [1] and adj[4] == [3], '[TC10] adjacency FAILED'

# ---- TC11: rcm_ordering 对简单链返回有效排列 ----
perm = rcm_ordering(adj, 5)
assert len(perm) == 5 and set(perm) == set(range(5)), '[TC11] rcm_ordering FAILED'

# ---- TC12: create_nif_beam_geometry 数量正确 ----
beams = create_nif_beam_geometry(num_cones=4, beams_per_cone=48)
assert len(beams) == 192, '[TC12] beam 数量 FAILED'

# ---- TC13: laser_beam_characteristics 均值合理 ----
stats = laser_beam_characteristics(beams)
assert stats['num_beams'] == 192 and 20.0 < stats['mean_polar_angle_deg'] < 60.0, '[TC13] beam stats FAILED'

# ---- TC14: compute_deposition_profile 输出有限且长度正确 ----
r_grid = np.linspace(0.0, TP.R_ABLATION, 51)
dep = compute_deposition_profile(beams, r_grid)
assert np.all(np.isfinite(dep)) and len(dep) == 50, '[TC14] deposition profile FAILED'

# ---- TC15: critical_density 正确数量级 ----
from laser_propagation import critical_density
nc = critical_density(351.0e-9)
assert 1.0e27 < nc < 1.0e28, '[TC15] critical_density FAILED'

# ---- TC16: plasma_refractive_index n_e < n_c 返回正实数 ----
from laser_propagation import plasma_refractive_index
n_idx = plasma_refractive_index(nc * 0.5, nc)
assert 0.0 < n_idx < 1.0, '[TC16] plasma_refractive_index FAILED'

# ---- TC17: spitzer_harm_conductivity 正输入返回正值 ----
from heat_conduction import spitzer_harm_conductivity
K = spitzer_harm_conductivity(1.0e6, 1.0, 1.0e26)
assert K > 0.0 and np.isfinite(K), '[TC17] spitzer_harm_conductivity FAILED'

# ---- TC18: compute_fusion_rate_density 低温接近0 ----
n_d = np.full(10, 1.0e28)
n_t = n_d.copy()
T_i = np.full(10, 100.0)
rate = compute_fusion_rate_density(n_d, n_t, T_i)
assert np.all(rate >= 0.0) and np.all(np.isfinite(rate)), '[TC18] fusion_rate_density 低温 FAILED'

# ---- TC19: alpha_deposition_local 线性比例验证 ----
rate_test = np.array([1.0e20, 2.0e20])
dep1 = alpha_deposition_local(rate_test, np.ones(2))
assert abs(dep1[1] / dep1[0] - 2.0) < 1.0e-6, '[TC19] alpha_deposition_local FAILED'

# ---- TC20: spitzer_equilibration_time 正输入返回有限值 ----
tau = spitzer_equilibration_time(1.0e26, 1.0e6, 1.0, 2.5)
assert tau > 0.0 and np.isfinite(tau), '[TC20] spitzer_equilibration_time FAILED'

# ---- TC21: apply_energy_relaxation 能量守恒 ----
E_ion = 1.0e9
E_e = 2.0e9
E_i_new, E_e_new = apply_energy_relaxation(E_ion, E_e, 1.0e-15, 1.0e26, 1.0e6, 1.0, 2.5)
assert abs((E_i_new + E_e_new) - (E_ion + E_e)) < 1.0e-6 * (E_ion + E_e), '[TC21] energy_relaxation FAILED'

# ---- TC22: build_energy_flow_digraph 返回8节点 ----
adj_m, names = build_energy_flow_digraph()
assert len(names) == 8 and adj_m.shape == (8, 8), '[TC22] digraph 尺寸 FAILED'

# ---- TC23: energy_flow_pagerank PR和为1 ----
pr = energy_flow_pagerank(adj_m)
assert abs(np.sum(pr) - 1.0) < 1.0e-6, '[TC23] pagerank 归一化 FAILED'

# ---- TC24: generate_surface_perturbation_ifs 输出长度等于模式数 ----
np.random.seed(42)
pert = generate_surface_perturbation_ifs(n_points=500, amplitude=1.0e-7, mode=12)
assert len(pert) == 12, '[TC24] perturbation 长度 FAILED'

# ---- TC25: RKF45Integrator 对常数函数保持解不变 ----
integrator = RKF45Integrator()
def f_const(t, y):
    return np.zeros_like(y)
y0 = np.array([1.0, 2.0])
y_new, t_new, h_new, accepted = integrator.step(f_const, 0.0, y0, 1.0e-12)
assert accepted and np.allclose(y_new, y0, atol=1.0e-6), '[TC25] RKF45 常数函数 FAILED'

# ---- TC26: NeutronMC.neutron_mean_free_path 返回正值 ----
mc = NeutronMC(n_samples=100)
mfp = mc.neutron_mean_free_path(250.0)
assert mfp > 0.0 and np.isfinite(mfp), '[TC26] neutron_mean_free_path FAILED'

# ---- TC27: LagrangeHydro 初始化密度为正 ----
mesh2 = RadialMesh(n_cells=20)
hydro = LagrangeHydro(mesh2)
assert np.all(hydro.rho > 0.0) and hydro.mesh.n_cells >= 20, '[TC27] LagrangeHydro 初始化 FAILED'

# ---- TC28: compute_laser_deposition_1d 零功率返回零 ----
r_cells = np.linspace(0.0, TP.R_ABLATION, 21)
r_nodes = np.linspace(0.0, TP.R_ABLATION, 22)
rho_dummy = np.full(21, 100.0)
T_dummy = np.full(21, 1000.0)
Z_dummy = np.full(21, 1.0)
dep_zero = compute_laser_deposition_1d(r_cells, r_nodes, rho_dummy, T_dummy, Z_dummy, 0.0, 0.0)
assert np.all(dep_zero == 0.0), '[TC28] laser_deposition 零功率 FAILED'

# ---- TC29: laser_power_time 负时间返回0 ----
P_neg = laser_power_time(-1.0e-9)
assert P_neg == 0.0, '[TC29] laser_power_time 负时间 FAILED'

# ---- TC30: compute_mode_growth_spectrum 返回字典 ----
rho_prof = np.full(20, 1000.0)
r_c = np.linspace(0.1e-3, 1.0e-3, 20)
u_n = np.zeros(21)
spectrum = compute_mode_growth_spectrum(rho_prof, r_c, u_n, mode_range=range(1, 6))
assert isinstance(spectrum, dict), '[TC30] mode_growth_spectrum FAILED'
