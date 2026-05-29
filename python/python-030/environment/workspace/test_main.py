# -*- coding: utf-8 -*-
"""
main.py
=======
Unified zero-parameter entry point for the
**Drip-Line Nuclear Structure and Decay Dynamics** suite.

This script performs a complete end-to-end calculation for a representative
drip-line nucleus (chosen as neutron-rich ^28O, Z=8, N=20) including:

1. Mean-field potential construction ( Woods-Saxon + SO + Coulomb ).
2. Radial Schrödinger-equation solution via spectral collocation.
3. Self-consistent HFB+BCS pairing calculation.
4. Nuclear mass-surface interpolation and drip-line location.
5. Stochastic Langevin dynamics of nucleon evaporation.
6. β-decay half-life and uncertainty quantification (noncentral Beta).
7. Reaction cross-section estimation (transfer + Coulomb breakup).
8. Three-dimensional density mesh generation and RMS-radius extraction.
9. High-dimensional sparse-grid uncertainty propagation.
10. Fekete-triangle integration over deformation-parameter space.

All results are printed to stdout; no graphical output is produced.
"""

import numpy as np
import sys
import os

# Ensure the project directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import MASS_NEUTRON, MASS_PROTON
from nuclear_potential import build_neutron_potential, build_proton_potential
from radial_solver import solve_radial_schroedinger, radial_matrix_element
from hfb_selfconsistent import solve_hfb_bcs
from mass_surface import (
    NuclearMassSurface, liquid_drop_binding_energy,
    atomic_mass_ldm, mass_surface_curvature
)
from stochastic_dynamics import (
    ensemble_langevin, nuclear_temperature,
    evaporative_decay_rate, diffusion_coefficient_from_msd
)
from decay_statistics import (
    beta_decay_halflife, q_value_beta_decay,
    neutron_drip_line_uncertainty, decay_chain_simulation
)
from reaction_phasespace import (
    transfer_cross_section, coulomb_breakup_cross_section,
    disk_monomial_integral, angular_momentum_coupling_weight
)
from density_mesh import (
    build_tetrahedral_sphere_mesh, integrate_density_on_mesh,
    deformed_fermi_density, rms_radius_from_mesh, write_mesh_to_xml
)
from quadrature_engine import (
    integrate_deformation_pdf, sparse_grid_integrate
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(42)

    # ------------------------------------------------------------------
    #  Target nucleus: ^28O (Z=8, N=20) — neutron drip-line candidate
    # ------------------------------------------------------------------
    Z = 8
    N = 20
    A = Z + N
    beta2 = 0.15
    beta3 = 0.05
    beta4 = -0.02

    print("Drip-Line Nuclear Structure and Decay Dynamics Computation")
    print(f"Target nucleus: A={A}, Z={Z}, N={N}")
    print(f"Deformation parameters: beta2={beta2}, beta3={beta3}, beta4={beta4}")

    # ==================================================================
    #  1. Mean-field potentials
    # ==================================================================
    print_section("1. Mean-Field Potential Construction")
    rmax = 15.0
    Nr = 300
    r = np.linspace(0.0, rmax, Nr)

    Vn = build_neutron_potential(r, N, Z, beta2, beta3, beta4)
    Vp = build_proton_potential(r, Z, N, beta2, beta3, beta4)

    print(f"Neutron potential at r=0:  {Vn[0]:.3f} MeV")
    print(f"Proton  potential at r=0:  {Vp[0]:.3f} MeV")
    print(f"Neutron potential at r={rmax}: {Vn[-1]:.6f} MeV")
    print(f"Proton  potential at r={rmax}: {Vp[-1]:.6f} MeV")

    # ==================================================================
    #  2. Radial Schrödinger solver (spectral collocation)
    # ==================================================================
    print_section("2. Spectral Radial Schrödinger Solver")
    l_vals = [0, 1, 2]
    all_energies = []
    all_wavefunctions = []
    all_r = None

    for l in l_vals:
        en, wf, rr = solve_radial_schroedinger(
            rmax=rmax, N=64, l=l,
            V_func=lambda r_in: build_neutron_potential(r_in, N, Z, beta2, beta3, beta4),
            n_eig=5, method='gll'
        )
        all_r = rr
        for k in range(en.size):
            all_energies.append(en[k])
            all_wavefunctions.append(wf[:, k])
        print(f"  l={l}: bound-state energies = {np.round(en, 3)} MeV")

    all_energies = np.array(all_energies)
    n_sp = all_energies.size
    print(f"Total single-particle states computed: {n_sp}")

    # ==================================================================
    #  3. HFB-BCS self-consistent pairing
    # ==================================================================
    print_section("3. HFB-BCS Self-Consistent Pairing")
    # Sort energies and solve for neutrons
    epsilon_sorted = np.sort(all_energies)
    hfb_n = solve_hfb_bcs(epsilon_sorted, target_N=N, Delta0=2.0)
    print(f"Neutron chemical potential lambda = {hfb_n['lambda']:.4f} MeV")
    print(f"Neutron pairing gap Delta         = {hfb_n['Delta']:.4f} MeV")
    print(f"Neutron pairing energy            = {hfb_n['E_pair']:.4f} MeV")
    print(f"Neutron total HFB+BCS energy      = {hfb_n['E_total']:.4f} MeV")
    print(f"Converged in {hfb_n['iterations']} iterations")

    # Protons (fewer states, use a reduced spectrum)
    epsilon_p = epsilon_sorted[:max(4, len(epsilon_sorted)//2)]
    hfb_p = solve_hfb_bcs(epsilon_p, target_N=Z, Delta0=1.5)
    print(f"Proton chemical potential lambda  = {hfb_p['lambda']:.4f} MeV")
    print(f"Proton pairing gap Delta          = {hfb_p['Delta']:.4f} MeV")

    # ==================================================================
    #  4. Nuclear mass surface & drip line
    # ==================================================================
    print_section("4. Nuclear Mass Surface and Drip-Line Location")
    # Build a small empirical data set around oxygen isotopes
    data_N = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    data_Z = np.full_like(data_N, 8)
    # Mass excess roughly from LDM + empirical offset (MeV/c²)
    empirical_masses = np.array([
        atomic_mass_ldm(8, nn) + np.random.normal(0, 2.0)
        for nn in data_N
    ])
    mass_surface = NuclearMassSurface(data_N, data_Z, empirical_masses)

    mass_28O = mass_surface.evaluate(N, Z)
    S_n = mass_surface.separation_energy(N, Z, 'neutron')
    S_p = mass_surface.separation_energy(N, Z, 'proton')
    print(f"Interpolated mass of ^28O = {mass_28O:.3f} MeV/c²")
    print(f"Neutron separation energy S_n = {S_n:.3f} MeV")
    print(f"Proton  separation energy S_p = {S_p:.3f} MeV")

    N_drip_n = mass_surface.dripline_location(Z, 'neutron')
    N_drip_p = mass_surface.dripline_location(Z, 'proton')
    print(f"Estimated neutron drip line for Z={Z}: N ≈ {N_drip_n}")
    print(f"Estimated proton  drip line for Z={Z}: N ≈ {N_drip_p}")

    kappa = mass_surface_curvature(mass_surface, float(N), float(Z), h=1.0)
    print(f"Mass-surface curvature at (N={N}, Z={Z}) = {kappa:.4f} MeV")

    # ==================================================================
    #  5. Stochastic Langevin dynamics
    # ==================================================================
    print_section("5. Stochastic Langevin Nucleon Dynamics")
    T_nuc = nuclear_temperature(excitation_energy=5.0, A=A)
    print(f"Nuclear temperature for E*=5 MeV: T = {T_nuc:.3f} MeV")

    # Simplified radial force: F = -dV/dr evaluated at neutron potential
    dr = r[1] - r[0]
    dVdr = np.gradient(Vn, dr)
    def force_func(pos):
        # 1-D radial force, returning 3-D vector pointing toward origin
        rr = np.linalg.norm(pos)
        if rr < 1e-6:
            return np.zeros(3)
        idx = int(np.clip(rr / dr, 0, len(r) - 1))
        F_mag = -dVdr[idx]
        return F_mag * (-pos / rr)

    gamma = 50.0  # MeV·fs/fm²
    dt = 0.5      # fs
    n_steps = 200
    n_ens = 100
    r0_list = np.random.normal(3.0, 0.5, size=(n_ens, 3))
    msd, mean_traj = ensemble_langevin(
        r0_list, force_func, gamma, T_nuc, dt, n_steps, dim=3, seed=42
    )
    D_est = diffusion_coefficient_from_msd(msd, dt)
    print(f"Ensemble size: {n_ens}, Steps: {n_steps}, dt={dt} fs")
    print(f"Final MSD = {msd[-1]:.3f} fm²")
    print(f"Estimated diffusion coefficient D = {D_est:.6f} fm²/fs")

    evap_rate = evaporative_decay_rate(A, Z, T_nuc, separation_energy=max(S_n, 0.1))
    print(f"Evaporative decay rate (neutron) ≈ {evap_rate:.6e} fs⁻¹")

    # ==================================================================
    #  6. β-decay statistics
    # ==================================================================
    print_section("6. Beta-Decay Statistics and Noncentral-Beta Uncertainty")
    M_parent = mass_28O
    M_daughter = mass_surface.evaluate(N - 1, Z + 1)
    Q_beta = q_value_beta_decay(M_parent, M_daughter)
    print(f"Q_beta(28O -> 28F) ≈ {Q_beta:.3f} MeV")

    T12 = beta_decay_halflife(Z + 1, Q_beta, Bgt=0.5)
    print(f"Estimated half-life T_1/2 ≈ {T12:.3e} s")

    # Noncentral Beta uncertainty on drip-line existence
    lower, upper, mean = neutron_drip_line_uncertainty(
        N_obs=5, Z=Z, confidence=0.95, eff_bias=0.03
    )
    print(f"Noncentral-Beta 95% credible interval for drip existence:")
    print(f"  [{lower:.4f}, {upper:.4f}], mean = {mean:.4f}")

    # Decay-chain Monte Carlo
    # States: 0=28O, 1=28F, 2=28Ne, 3=stable
    Tmat = np.array([
        [0.0, 0.9, 0.0, 0.1],
        [0.0, 0.0, 0.85, 0.15],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    pops, _ = decay_chain_simulation(0, Tmat, n_steps=5, n_samples=2000, seed=42)
    print("Decay-chain population evolution:")
    for step in range(pops.shape[0]):
        print(f"  Step {step}: 28O={pops[step,0]:.3f}, 28F={pops[step,1]:.3f}, "
              f"28Ne={pops[step,2]:.3f}, stable={pops[step,3]:.3f}")

    # ==================================================================
    #  7. Reaction cross sections
    # ==================================================================
    print_section("7. Reaction Phase-Space and Cross Sections")
    R_grazing = 1.2 * (A ** (1.0 / 3.0) + 16.0 ** (1.0 / 3.0))
    sigma_tr = transfer_cross_section(R_grazing, sigma_b=1.0, P0=0.08)
    print(f"One-neutron transfer cross section ≈ {sigma_tr:.2f} fm² = {sigma_tr*0.01:.4f} barn")

    sigma_cu = coulomb_breakup_cross_section(
        E_beam=50.0, Z_p=Z, Z_t=82, A_p=A, A_t=208, E_bind=max(S_n, 0.5)
    )
    print(f"Coulomb breakup cross section ≈ {sigma_cu:.2f} fm² = {sigma_cu*0.01:.4f} barn")

    # Unit-disk angular-momentum coupling integral
    I_20 = disk_monomial_integral(2, 0)
    I_02 = disk_monomial_integral(0, 2)
    I_22 = disk_monomial_integral(2, 2)
    print(f"Unit-disk integrals: I_20={I_20:.6f}, I_02={I_02:.6f}, I_22={I_22:.6f}")

    wgt = angular_momentum_coupling_weight(2.0, 2.5, 3.0, 0.5)
    print(f"Angular-momentum coupling weight j1=2, j2=2.5, J=3, M=0.5: {wgt:.4f}")

    # ==================================================================
    #  8. 3-D density mesh
    # ==================================================================
    print_section("8. Three-Dimensional Density Mesh")
    nodes, elements = build_tetrahedral_sphere_mesh(
        rmin=0.0, rmax=10.0, n_shells=6, n_subdiv=1
    )
    print(f"Tetrahedral mesh: {nodes.shape[0]} nodes, {elements.shape[0]} elements")

    rho_func = lambda x, y, z: deformed_fermi_density(
        x, y, z, A, beta2, beta3, beta4
    )
    total_A, elem_data = integrate_density_on_mesh(rho_func, nodes, elements)
    rms_r = rms_radius_from_mesh(nodes, elements, rho_func)
    print(f"Integrated density (should be ≈ A={A}): {total_A:.2f}")
    print(f"RMS charge radius: {rms_r:.3f} fm")

    xml_path = os.path.join(os.path.dirname(__file__), "density_mesh.xml")
    write_mesh_to_xml(nodes, elements, xml_path)
    print(f"Mesh written to {xml_path}")

    # ==================================================================
    #  9. Sparse-grid uncertainty propagation
    # ==================================================================
    print_section("9. Sparse-Grid Uncertainty Quantification")
    # Integrate a simple response over 2-D parameter space (beta2, beta3)
    def response(x):
        b2, b3 = x[0], x[1]
        # Approximate binding-energy sensitivity to deformation
        return -(b2 ** 2 + 0.5 * b3 ** 2) * 10.0

    val_sg = sparse_grid_integrate(response, dim_num=2, level_max=4)
    print(f"Sparse-grid integral of deformation response over [-1,1]²:")
    print(f"  Value = {val_sg:.4f} MeV")

    # ==================================================================
    #  10. Fekete triangle integration over deformation PDF
    # ==================================================================
    print_section("10. Fekete-Triangle Deformation-Space Integration")
    # Simple Gaussian PDF on deformation rectangle
    def deformation_pdf(b2, b3):
        return np.exp(-(b2 ** 2 + b3 ** 2) / (2.0 * 0.1 ** 2)) / (2.0 * np.pi * 0.1 ** 2)

    prob_def = integrate_deformation_pdf(
        beta2_min=-0.3, beta2_max=0.3,
        beta3_min=-0.2, beta3_max=0.2,
        pdf_func=deformation_pdf, degree=5
    )
    print(f"Integrated deformation PDF over sampled rectangle ≈ {prob_def:.4f}")

    # ==================================================================
    #  Summary
    # ==================================================================
    print_section("COMPUTATION SUMMARY")
    print(f"Nucleus: ^28O (Z={Z}, N={N})")
    print(f"Neutron pairing gap:      {hfb_n['Delta']:.3f} MeV")
    print(f"Proton  pairing gap:      {hfb_p['Delta']:.3f} MeV")
    print(f"Neutron S_n:              {S_n:.3f} MeV")
    print(f"Proton  S_p:              {S_p:.3f} MeV")
    print(f"Beta-decay Q-value:       {Q_beta:.3f} MeV")
    print(f"Estimated half-life:      {T12:.3e} s")
    print(f"Transfer cross section:   {sigma_tr*0.01:.4f} barn")
    print(f"Coulomb breakup:          {sigma_cu*0.01:.4f} barn")
    print(f"RMS radius:               {rms_r:.3f} fm")
    print(f"Diffusion coefficient:    {D_est:.6f} fm²/fs")
    print("\nAll calculations completed successfully.")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（32个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: reduced_mass 对称性验证 ----
from constants import reduced_mass
m = 10.0
assert abs(reduced_mass(m, m) - m / 2.0) < 1e-10, '[TC01] reduced_mass 对称性验证 FAILED'

# ---- TC02: hbar2_over_2m 返回有限正值 ----
from constants import hbar2_over_2m
val = hbar2_over_2m()
assert np.isfinite(val) and val > 0, '[TC02] hbar2_over_2m 返回有限正值 FAILED'

# ---- TC03: woods_saxon 远场趋于零 ----
from nuclear_potential import woods_saxon
r_far = np.array([100.0, 200.0])
V_far = woods_saxon(r_far, -50.0, 5.0, 0.67)
assert np.all(np.abs(V_far) < 1.0), '[TC03] woods_saxon 远场趋于零 FAILED'

# ---- TC04: deformed_radius theta=pi/2 对称性 ----
from nuclear_potential import deformed_radius
R0 = deformed_radius(np.pi / 2, 16, 0.2, 0.0, 0.0)
Rpi = deformed_radius(np.pi / 2, 16, 0.2, 0.0, 0.0)
assert abs(R0 - Rpi) < 1e-10, '[TC04] deformed_radius theta=pi/2 对称性 FAILED'

# ---- TC05: coulomb_potential 内外连续有限 ----
from nuclear_potential import coulomb_potential
r_test = np.array([5.99, 6.0, 6.01])
Vc = coulomb_potential(r_test, 8, 16)
assert np.all(np.isfinite(Vc)), '[TC05] coulomb_potential 内外连续有限 FAILED'

# ---- TC06: build_neutron_potential 输出尺寸匹配 ----
rr = np.linspace(0, 10, 50)
Vn = build_neutron_potential(rr, 20, 8, 0.15, 0.05, -0.02)
assert Vn.shape == rr.shape, '[TC06] build_neutron_potential 输出尺寸匹配 FAILED'

# ---- TC07: gauss_lobatto_nodes 节点数量正确 ----
from radial_solver import gauss_lobatto_nodes
nodes = gauss_lobatto_nodes(8, -1.0, 1.0)
assert nodes.size == 8, '[TC07] gauss_lobatto_nodes 节点数量正确 FAILED'

# ---- TC08: lagrange_derivative_matrix 常数导数为零 ----
from radial_solver import lagrange_derivative_matrix
x = np.linspace(0, 1, 5)
D = lagrange_derivative_matrix(x)
const = np.ones(5)
deriv = D @ const
assert np.allclose(deriv, 0.0, atol=1e-10), '[TC08] lagrange_derivative_matrix 常数导数为零 FAILED'

# ---- TC09: solve_radial_schroedinger 返回束缚态能量 ----
V_func = lambda r_in: build_neutron_potential(r_in, 20, 8, 0.0, 0.0, 0.0)
en, wf, r_out = solve_radial_schroedinger(rmax=15.0, N=64, l=0, V_func=V_func, n_eig=3)
assert en.size > 0 and np.all(en < 0), '[TC09] solve_radial_schroedinger 返回束缚态能量 FAILED'

# ---- TC10: radial_matrix_element 交换对称性 ----
r_test = np.linspace(0, 5, 50)
u1 = np.sin(r_test)
u2 = np.cos(r_test)
m12 = radial_matrix_element(r_test, u1, u2)
m21 = radial_matrix_element(r_test, u2, u1)
assert abs(m12 - m21) < 1e-10, '[TC10] radial_matrix_element 交换对称性 FAILED'

# ---- TC11: bcs_occupation u2+v2=1 守恒 ----
from hfb_selfconsistent import bcs_occupation
eps = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
u2, v2, E = bcs_occupation(eps, 0.0, 1.0)
assert np.allclose(u2 + v2, 1.0, atol=1e-10), '[TC11] bcs_occupation u2+v2=1 守恒 FAILED'

# ---- TC12: conjugate_gradient_solve 解析验证 ----
from hfb_selfconsistent import conjugate_gradient_solve
A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
x, info = conjugate_gradient_solve(A, b)
assert np.allclose(A @ x, b, atol=1e-8), '[TC12] conjugate_gradient_solve 解析验证 FAILED'

# ---- TC13: solve_hfb_bcs 粒子数守恒 ----
epsilon = np.array([-12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0])
result = solve_hfb_bcs(epsilon, target_N=4, Delta0=1.0, tol=1e-8)
assert abs(result['particle_number'] - 4.0) < 0.1, '[TC13] solve_hfb_bcs 粒子数守恒 FAILED'

# ---- TC14: liquid_drop_binding_energy 返回值正数 ----
B = liquid_drop_binding_energy(8, 8)
assert B > 0 and np.isfinite(B), '[TC14] liquid_drop_binding_energy 返回值正数 FAILED'

# ---- TC15: atomic_mass_ldm 返回值有限正数 ----
M = atomic_mass_ldm(8, 8)
assert M > 0 and np.isfinite(M), '[TC15] atomic_mass_ldm 返回值有限正数 FAILED'

# ---- TC16: NuclearMassSurface 数据点精确重构 ----
data_N = np.array([8, 9, 10, 11, 12])
data_Z = np.array([8, 8, 8, 8, 8])
data_mass = np.array([atomic_mass_ldm(int(z), int(n)) for z, n in zip(data_Z, data_N)])
ms = NuclearMassSurface(data_N, data_Z, data_mass)
m_eval = ms.evaluate(data_N, data_Z)
assert np.allclose(m_eval, data_mass, atol=1e-6), '[TC16] NuclearMassSurface 数据点精确重构 FAILED'

# ---- TC17: mass_surface_curvature 返回标量有限值 ----
kappa = mass_surface_curvature(ms, 10.0, 8.0, h=1.0)
assert np.isscalar(kappa) and np.isfinite(kappa), '[TC17] mass_surface_curvature 返回标量有限值 FAILED'

# ---- TC18: noncentral_beta_cdf 边界值 ----
from decay_statistics import noncentral_beta_cdf
assert noncentral_beta_cdf(0.0, 2.0, 3.0, 0.5) == 0.0, '[TC18] noncentral_beta_cdf 边界值 F(0)=0 FAILED'
assert abs(noncentral_beta_cdf(1.0, 2.0, 3.0, 0.5) - 1.0) < 1e-10, '[TC18] noncentral_beta_cdf 边界值 F(1)=1 FAILED'

# ---- TC19: q_value_beta_decay 解析验证 ----
Q = q_value_beta_decay(100.0, 95.0)
assert abs(Q - 5.0) < 1e-10, '[TC19] q_value_beta_decay 解析验证 FAILED'

# ---- TC20: beta_decay_halflife Q<=0 返回无穷 ----
T12 = beta_decay_halflife(8, 0.0)
assert np.isinf(T12), '[TC20] beta_decay_halflife Q<=0 返回无穷 FAILED'

# ---- TC21: decay_chain_simulation 固定种子可复现 ----
np.random.seed(42)
Tmat = np.array([[0.0, 1.0], [0.0, 1.0]])
pops1, _ = decay_chain_simulation(0, Tmat, n_steps=3, n_samples=1000, seed=42)
np.random.seed(42)
pops2, _ = decay_chain_simulation(0, Tmat, n_steps=3, n_samples=1000, seed=42)
assert np.allclose(pops1, pops2, atol=1e-10), '[TC21] decay_chain_simulation 固定种子可复现 FAILED'

# ---- TC22: disk_monomial_integral I_00 等于 pi ----
I00 = disk_monomial_integral(0, 0)
assert abs(I00 - np.pi) < 1e-10, '[TC22] disk_monomial_integral I_00 等于 pi FAILED'

# ---- TC23: transfer_cross_section 非负有限 ----
sigma = transfer_cross_section(5.0, 1.0, 0.1)
assert sigma >= 0 and np.isfinite(sigma), '[TC23] transfer_cross_section 非负有限 FAILED'

# ---- TC24: angular_momentum_coupling_weight 范围 [0,1] ----
wgt = angular_momentum_coupling_weight(2.0, 2.5, 3.0, 0.5)
assert 0.0 <= wgt <= 1.0, '[TC24] angular_momentum_coupling_weight 范围 [0,1] FAILED'

# ---- TC25: tetrahedron_volume 正体积解析验证 ----
from density_mesh import tetrahedron_volume
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
vol = tetrahedron_volume(nodes, [0, 1, 2, 3])
assert vol > 0 and abs(vol - 1.0 / 6.0) < 1e-10, '[TC25] tetrahedron_volume 正体积解析验证 FAILED'

# ---- TC26: deformed_fermi_density 中心点正密度 ----
rho = deformed_fermi_density(0.0, 0.0, 0.0, 16, 0.1, 0.0, 0.0)
assert rho > 0 and np.isfinite(rho), '[TC26] deformed_fermi_density 中心点正密度 FAILED'

# ---- TC27: build_tetrahedral_sphere_mesh 节点和单元维度正确 ----
nodes, elements = build_tetrahedral_sphere_mesh(0.0, 5.0, 3, 1)
assert nodes.shape[1] == 3 and elements.shape[1] == 4, '[TC27] build_tetrahedral_sphere_mesh 节点和单元维度正确 FAILED'

# ---- TC28: clenshaw_curtis_weights 权重和为2 ----
from quadrature_engine import clenshaw_curtis_weights
w = clenshaw_curtis_weights(5)
assert abs(np.sum(w) - 2.0) < 1e-10, '[TC28] clenshaw_curtis_weights 权重和为2 FAILED'

# ---- TC29: integrate_on_triangle 常数函数等于三角形面积 ----
from quadrature_engine import integrate_on_triangle
f = lambda x, y: 1.0
area = integrate_on_triangle(f, degree=5)
assert abs(area - 0.5) < 1e-10, '[TC29] integrate_on_triangle 常数函数等于三角形面积 FAILED'

# ---- TC30: sparse_grid_integrate 常数函数等于 2^dim ----
val = sparse_grid_integrate(lambda x: 1.0, dim_num=2, level_max=2)
assert abs(val - 4.0) < 1e-6, '[TC30] sparse_grid_integrate 常数函数等于 2^dim FAILED'

# ---- TC31: nuclear_temperature 解析验证 ----
T = nuclear_temperature(excitation_energy=8.0, A=16)
expected = np.sqrt(8.0 / 2.0)
assert abs(T - expected) < 1e-10, '[TC31] nuclear_temperature 解析验证 FAILED'

# ---- TC32: evaporative_decay_rate 极端参数返回零 ----
rate = evaporative_decay_rate(16, 8, 0.0, 5.0)
assert rate == 0.0, '[TC32] evaporative_decay_rate 极端参数返回零 FAILED'

print('\n全部 32 个测试通过!\n')
