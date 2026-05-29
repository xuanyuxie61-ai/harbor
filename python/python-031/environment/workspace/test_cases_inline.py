# ---- TC01: skyrme_energy_density 零密度返回零 ----
from nuclear_eos import skyrme_energy_density
ed, pr = skyrme_energy_density(0.0, 0.0)
assert ed == 0.0 and pr == 0.0, '[TC01] skyrme_energy_density 零密度 FAILED'

# ---- TC02: nuclear_matter_properties 饱和密度能量为有限负值 ----
props = nuclear_matter_properties(0.16, 0.5)
assert np.isfinite(props['energy_per_nucleon']), '[TC02] 能量非有限 FAILED'
assert props['energy_per_nucleon'] < 0, '[TC02] 能量非负 FAILED'

# ---- TC03: nuclear_matter_properties 对称能为正值 ----
props = nuclear_matter_properties(0.08, 0.3)
assert props['symmetry_energy'] > 0, '[TC03] 对称能非正 FAILED'

# ---- TC04: alnorm 正态CDF在x=0处为0.5 ----
from nuclear_eos import alnorm
val = alnorm(0.0, upper=True)
assert abs(val - 0.5) < 1e-10, '[TC04] alnorm(0,upper) FAILED'

# ---- TC05: tnc df<=0返回错误标志2 ----
from nuclear_eos import tnc
val, ifault = tnc(0.0, 0.0, 0.0)
assert ifault == 2, '[TC05] tnc df<=0 FAILED'

# ---- TC06: parameter_uncertainty_t_stat 覆盖率在[0,1]区间内 ----
cov, ci_l, ci_u = parameter_uncertainty_t_stat(np.array([-1800.0]), -1800.0, np.array([10.0]))
assert 0.0 <= cov <= 1.0, '[TC06] 覆盖率范围 FAILED'

# ---- TC07: create_pasta_phase 五种相S/V均为正 ----
for pid in range(1, 6):
    phase = create_pasta_phase(pid, 0.08, 0.3)
    assert phase.surface_to_volume() > 0, f'[TC07] Phase {pid} S/V FAILED'

# ---- TC08: PastaPhase.PHASE_NAMES 包含5个相 ----
assert len(PastaPhase.PHASE_NAMES) == 5, '[TC08] 相数量 FAILED'

# ---- TC09: triangle01_monomial_integral [0,0]等于0.5 ----
from geometry_pasta import triangle01_monomial_integral
val = triangle01_monomial_integral([0, 0])
assert abs(val - 0.5) < 1e-12, '[TC09] 单位三角形积分 FAILED'

# ---- TC10: tetrahedron_unit_monomial [0,0,0]等于1/6 ----
from geometry_pasta import tetrahedron_unit_monomial
val = tetrahedron_unit_monomial([0, 0, 0])
assert abs(val - 1.0 / 6.0) < 1e-12, '[TC10] 单位四面体积分 FAILED'

# ---- TC11: analytical_coulomb 五种相返回值非负 ----
for pid in range(1, 6):
    e_c = analytical_coulomb(pid, 0.08, 0.3)
    assert e_c >= 0, f'[TC11] Phase {pid} Coulomb FAILED'

# ---- TC12: beta_decay_rates 零温度返回零 ----
lp, lm = beta_decay_rates(0.0, 0.05, 0.02, 50.0)
assert lp == 0.0 and lm == 0.0, '[TC12] 零温度衰变率 FAILED'

# ---- TC13: diffusion_coefficient 正参数返回正值 ----
D = diffusion_coefficient(1.0, 1.0, 5.0)
assert D > 0, '[TC13] 扩散系数非正 FAILED'

# ---- TC14: fd_reaction_diffusion_1d 总核子数近似守恒 ----
N = 20
x = np.linspace(0, 5, N)
rho_n0 = np.ones(N) * 0.05
rho_p0 = np.ones(N) * 0.02
total0 = np.sum(rho_n0 + rho_p0)
rho_n, rho_p, _, _ = fd_reaction_diffusion_1d(
    rho_n0, rho_p0, dx=x[1] - x[0], dt=0.001, n_steps=50,
    D_n=0.1, D_p=0.1, lambda_plus=0.01, lambda_minus=0.01, bc_type='neumann'
)
total1 = np.sum(rho_n + rho_p)
assert abs(total1 - total0) / total0 < 0.1, '[TC14] 总核子数守恒 FAILED'

# ---- TC15: unstable_exact t=0时等于初始条件[1, mu] ----
from ode_integrator import unstable_exact
y1, y2 = unstable_exact(0.0, 0.5)
assert abs(y1 - 1.0) < 1e-12 and abs(y2 - 0.5) < 1e-12, '[TC15] unstable_exact初值 FAILED'

# ---- TC16: unstable_deriv mu=1时导数计算正确 ----
from ode_integrator import unstable_deriv
dydt = unstable_deriv(0.0, [1.0, 0.0], 1.0)
assert abs(dydt[0] - 1.0) < 1e-12 and abs(dydt[1] + 1.0) < 1e-12, '[TC16] unstable_deriv FAILED'

# ---- TC17: tough_deriv 输出长度为4 ----
from ode_integrator import tough_deriv
dydt = tough_deriv(0.0, [1.0, 1.0, 0.0, 1.0])
assert len(dydt) == 4, '[TC17] tough_deriv 维度 FAILED'

# ---- TC18: neutrino_luminosity 零温度返回0 ----
from ode_integrator import neutrino_luminosity
eps = neutrino_luminosity(0.0, 0.1, 0.3)
assert eps == 0.0, '[TC18] 零温度中微子发光度 FAILED'

# ---- TC19: heat_capacity_degenerate 正温度正比热 ----
from ode_integrator import heat_capacity_degenerate
cv = heat_capacity_degenerate(0.1, 0.3, 1.0)
assert cv > 0, '[TC19] 比热非正 FAILED'

# ---- TC20: besselj_zero J0第一个零点约2.4048 ----
zeros = besselj_zero(0, 3)
assert abs(zeros[0] - 2.4048255577) < 1e-4, '[TC20] J0第一个零点 FAILED'

# ---- TC21: spherical_coulomb_potential 球内递减球外衰减 ----
r = np.array([0.0, 0.5, 1.0, 2.0])
R_ws = (3.0 / (4.0 * np.pi * 0.08)) ** (1.0 / 3.0)
phi = spherical_coulomb_potential(r, R_ws, 0.08 * 0.3)
assert phi[0] > phi[2], '[TC21] 球内势递减 FAILED'
assert phi[-1] < phi[0], '[TC21] 球外势衰减 FAILED'

# ---- TC22: cylinder_vibration_frequencies 正参数返回非空递增序列 ----
freqs = cylinder_vibration_frequencies(5.0, 1.0, 0.08 * 939.0)
assert len(freqs) > 0, '[TC22] 振动频率为空 FAILED'
assert np.all(np.diff(freqs) >= 0), '[TC22] 频率非递增 FAILED'

# ---- TC23: monte_carlo_nd_integral 2D高斯积分接近理论值pi ----
np.random.seed(42)
integral, error = monte_carlo_nd_integral(nd_integrand_gaussian, 2, -3, 3, n_samples=50000)
exact = np.pi
assert abs(integral - exact) / exact < 0.05, '[TC23] 2D高斯积分 FAILED'

# ---- TC24: nd_integrand_gaussian 在零点值为1 ----
x = np.zeros((2, 5))
val = nd_integrand_gaussian(2, 5, x)
assert np.allclose(val, 1.0), '[TC24] 高斯被积函数零点 FAILED'

# ---- TC25: total_energy_per_nucleon 含bulk分量 ----
E_total, comp = total_energy_per_nucleon(1, 0.08, 0.3)
assert 'bulk' in comp and np.isfinite(comp['bulk']), '[TC25] bulk分量缺失 FAILED'

# ---- TC26: optimal_filling 返回u在(0,1)区间内 ----
u_opt, E_min = optimal_filling(1, 0.08, 0.3)
assert 0.0 < u_opt < 1.0, '[TC26] 最优填充率范围 FAILED'

# ---- TC27: surface_tension 对称物质为正 ----
from phase_diagram import surface_tension
sigma = surface_tension(0.16, 0.5)
assert sigma > 0, '[TC27] 表面张力非正 FAILED'

# ---- TC28: lattice_energy 为负值 ----
from phase_diagram import lattice_energy
E_lat = lattice_energy(0.08, 0.3)
assert E_lat < 0, '[TC28] 晶格能非负 FAILED'

# ---- TC29: pasta_deformation_energy m=2柱相为正 ----
from bessel_modes import pasta_deformation_energy
dE = pasta_deformation_energy(2, 5.0, 0.1, 2, 1.0)
assert dE > 0, '[TC29] 柱相形变能非正 FAILED'

# ---- TC30: transition_density 搜索返回值在合理范围 ----
rho_t, found = transition_density(1, 2, 0.3, rho_min=0.02, rho_max=0.15)
if found:
    assert 0.02 <= rho_t <= 0.15, '[TC30] 转变密度范围 FAILED'
assert (found and rho_t is not None) or (not found and rho_t is None), '[TC30] 转变密度返回值 FAILED'
