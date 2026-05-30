# -*- coding: utf-8 -*-

import numpy as np
import sys
import os


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
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(42)




    Z = 8
    N = 20
    A = Z + N
    beta2 = 0.15
    beta3 = 0.05
    beta4 = -0.02

    print("Drip-Line Nuclear Structure and Decay Dynamics Computation")
    print(f"Target nucleus: A={A}, Z={Z}, N={N}")
    print(f"Deformation parameters: beta2={beta2}, beta3={beta3}, beta4={beta4}")




    print_section("1. Mean-Field Potential Construction")
    rmax = 15.0
    Nr = 300
    r = np.linspace(0.0, rmax, Nr)













    raise NotImplementedError("HOLE 3: mean-field construction and radial solver call chain is not implemented.")




    print_section("3. HFB-BCS Self-Consistent Pairing")

    epsilon_sorted = np.sort(all_energies)
    hfb_n = solve_hfb_bcs(epsilon_sorted, target_N=N, Delta0=2.0)
    print(f"Neutron chemical potential lambda = {hfb_n['lambda']:.4f} MeV")
    print(f"Neutron pairing gap Delta         = {hfb_n['Delta']:.4f} MeV")
    print(f"Neutron pairing energy            = {hfb_n['E_pair']:.4f} MeV")
    print(f"Neutron total HFB+BCS energy      = {hfb_n['E_total']:.4f} MeV")
    print(f"Converged in {hfb_n['iterations']} iterations")


    epsilon_p = epsilon_sorted[:max(4, len(epsilon_sorted)//2)]
    hfb_p = solve_hfb_bcs(epsilon_p, target_N=Z, Delta0=1.5)
    print(f"Proton chemical potential lambda  = {hfb_p['lambda']:.4f} MeV")
    print(f"Proton pairing gap Delta          = {hfb_p['Delta']:.4f} MeV")




    print_section("4. Nuclear Mass Surface and Drip-Line Location")

    data_N = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    data_Z = np.full_like(data_N, 8)

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




    print_section("5. Stochastic Langevin Nucleon Dynamics")
    T_nuc = nuclear_temperature(excitation_energy=5.0, A=A)
    print(f"Nuclear temperature for E*=5 MeV: T = {T_nuc:.3f} MeV")


    dr = r[1] - r[0]
    dVdr = np.gradient(Vn, dr)
    def force_func(pos):

        rr = np.linalg.norm(pos)
        if rr < 1e-6:
            return np.zeros(3)
        idx = int(np.clip(rr / dr, 0, len(r) - 1))
        F_mag = -dVdr[idx]
        return F_mag * (-pos / rr)

    gamma = 50.0
    dt = 0.5
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




    print_section("6. Beta-Decay Statistics and Noncentral-Beta Uncertainty")
    M_parent = mass_28O
    M_daughter = mass_surface.evaluate(N - 1, Z + 1)
    Q_beta = q_value_beta_decay(M_parent, M_daughter)
    print(f"Q_beta(28O -> 28F) ≈ {Q_beta:.3f} MeV")

    T12 = beta_decay_halflife(Z + 1, Q_beta, Bgt=0.5)
    print(f"Estimated half-life T_1/2 ≈ {T12:.3e} s")


    lower, upper, mean = neutron_drip_line_uncertainty(
        N_obs=5, Z=Z, confidence=0.95, eff_bias=0.03
    )
    print(f"Noncentral-Beta 95% credible interval for drip existence:")
    print(f"  [{lower:.4f}, {upper:.4f}], mean = {mean:.4f}")



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




    print_section("7. Reaction Phase-Space and Cross Sections")
    R_grazing = 1.2 * (A ** (1.0 / 3.0) + 16.0 ** (1.0 / 3.0))
    sigma_tr = transfer_cross_section(R_grazing, sigma_b=1.0, P0=0.08)
    print(f"One-neutron transfer cross section ≈ {sigma_tr:.2f} fm² = {sigma_tr*0.01:.4f} barn")

    sigma_cu = coulomb_breakup_cross_section(
        E_beam=50.0, Z_p=Z, Z_t=82, A_p=A, A_t=208, E_bind=max(S_n, 0.5)
    )
    print(f"Coulomb breakup cross section ≈ {sigma_cu:.2f} fm² = {sigma_cu*0.01:.4f} barn")


    I_20 = disk_monomial_integral(2, 0)
    I_02 = disk_monomial_integral(0, 2)
    I_22 = disk_monomial_integral(2, 2)
    print(f"Unit-disk integrals: I_20={I_20:.6f}, I_02={I_02:.6f}, I_22={I_22:.6f}")

    wgt = angular_momentum_coupling_weight(2.0, 2.5, 3.0, 0.5)
    print(f"Angular-momentum coupling weight j1=2, j2=2.5, J=3, M=0.5: {wgt:.4f}")




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




    print_section("9. Sparse-Grid Uncertainty Quantification")

    def response(x):
        b2, b3 = x[0], x[1]

        return -(b2 ** 2 + 0.5 * b3 ** 2) * 10.0

    val_sg = sparse_grid_integrate(response, dim_num=2, level_max=4)
    print(f"Sparse-grid integral of deformation response over [-1,1]²:")
    print(f"  Value = {val_sg:.4f} MeV")




    print_section("10. Fekete-Triangle Deformation-Space Integration")

    def deformation_pdf(b2, b3):
        return np.exp(-(b2 ** 2 + b3 ** 2) / (2.0 * 0.1 ** 2)) / (2.0 * np.pi * 0.1 ** 2)

    prob_def = integrate_deformation_pdf(
        beta2_min=-0.3, beta2_max=0.3,
        beta3_min=-0.2, beta3_max=0.2,
        pdf_func=deformation_pdf, degree=5
    )
    print(f"Integrated deformation PDF over sampled rectangle ≈ {prob_def:.4f}")




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
